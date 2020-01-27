#!/usr/bin/env python

import rospy
import numpy as np
from geometry_msgs.msg import Point
import GPy
import time

from sensor_msgs.msg import JointState

from std_msgs.msg import Float64
from bwrobot.srv import Jitter, MoveJoint, MoveJointRequest
from bwrobot.msg import LocalGP, OSGPR_GP, Arrays
import baxter_interface
import sys
sys.path.insert(0, '/home/bpwilcox/catkin_ws/src/SOLAR_GP-ROS/scripts/')
from data_buffer import DataBuffer
from trainer import Solar_Trainer

class BaxterTrainer(Solar_Trainer):
    """
    Baxter implementation of Solar_Trainer
    """
    def __init__(self, njit, degrees, num_inducing, wgen, use_old_Z = False):
        Solar_Trainer.__init__(self, njit, degrees, num_inducing, wgen, use_old_Z)        

    def setup_robot(self):
        arm = rospy.get_param('~arm', 'left')
        self.x_topic = 'robot/limb/' + arm + '/endpoint_state'
        self.y_topic = rospy.get_param('~y_topic', 'robot/joint_states')
        joints = rospy.get_param('~joints', ['s0', 's1', 'e1', 'w1'])
        self.joint_names = [arm + '_' + joint for joint in joints]
        self.limb = baxter_interface.Limb(arm)        
        return True
    
    def jitter_robot(self): 
        # Set robot to "default" position
        self.limb.move_to_neutral()
        YStart = []
        for joint in self.joint_names:
            YStart.append(self.limb.joint_angle(joint))

        YStart = np.array(YStart).reshape(1,len(self.joint_names))
        YI = self.jitter(self.njit, YStart, self.degrees)
        XI = np.empty([0,3])
        rospy.Rate(1).sleep()

        for y in YI:        
            self.limb.move_to_joint_positions(dict(zip(self.joint_names, y.tolist())))
            end_pose = self.limb.endpoint_pose()
            XI = np.vstack((XI,np.array([end_pose['position'].x, end_pose['position'].y, end_pose['position'].z]).reshape(1,3)))
        return XI, YI
        
def train():
    
    rospy.init_node('train_node')
    "Model Parameters"
    njit = rospy.get_param('~njit', 25)
    deg = rospy.get_param('~degree', 3)
    num_inducing = rospy.get_param('~num_inducing', 25)
    w_gen = rospy.get_param('~wgen', 0.975)
    
    Trainer = BaxterTrainer(njit, deg, num_inducing, w_gen, False)
    Trainer.initialize()
    Trainer.run()

if __name__ == '__main__':
    try:
        train()
    except rospy.ROSInterruptException:
        pass
