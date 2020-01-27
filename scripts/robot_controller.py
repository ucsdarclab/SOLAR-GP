#!/usr/bin/env python

import rospy
from bwrobot.msg import Arrays
from sensor_msgs.msg import JointState
from bwrobot.srv import Jitter, SetNeutral
from teleop_utils.srv import GetTeleop, GetTeleopResponse

class RobotController():
    """
    This base Controller class controls a robot to a commanded joint state from the Predictor.
    Special teleoperator commands can be triggered if implemented.
    """
    def __init__(self, joint_names, joint_topic):
        """
        joint_names: list of strings for joint names
        joint_topic: name of joint state topic
        """
        self.joint_names = joint_names
        self.nextY = list()
        self.currentY = list()
        self.joint_angles = dict()

        # Exposes a service for commanding robot to go to neutral position
        self.neutral_service = rospy.Service('set_neutral', SetNeutral, self.neutral_callback)
        # Exposes a service for commanding robot to jitter given a number of points and degree range
        self.jit_service= rospy.Service('jitter', Jitter, self.jitter_callback)

        rospy.Subscriber('prediction', Arrays, self.pred_callback) # Subscribes to current joint prediction
        rospy.Subscriber(joint_topic, JointState, self.curr_callback, queue_size = 10) # Subscribes to current joint state
        
    def neutral_callback(self, req):
        self.set_neutral()
        return True

    def jitter_callback(self, req):
        self.jitter(req.num_jit, req.degree)
        return True
    
    def pred_callback(self, msg):
        self.nextY = msg.array

    def curr_callback(self, msg):   
        for idx, name in enumerate(msg.name):
            if name in self.joint_names:
                self.joint_angles[name] = msg.position[idx]
        
        Y = []
        for joint in self.joint_names:
            Y.append(self.joint_angles[joint])
        self.currentY = Y

    def jitter(self, njit, deg):
        print("Jitter not implemented")
        return False

    def set_neutral(self):
        print("Set Neutral not implemented")
        return False

    def move_joints(self):
        print("Move Joints not implemented")
        return False

    def on_teleop(self, state):
        print("On Teleop not implemented")
        return False
    
    def run(self):
        rospy.wait_for_service('get_teleop')
        get_teleop = rospy.ServiceProxy('get_teleop', GetTeleop)
        teleop_state = GetTeleopResponse()
        
        rospy.wait_for_message('prediction',Arrays)
        # Check for on teleop state commands
        while not rospy.is_shutdown():
            teleop_state = get_teleop()
            if not self.on_teleop(teleop_state):
                continue
            # Send control commands to robot to move joints
            self.move_joints()