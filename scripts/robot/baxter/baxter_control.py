#!/usr/bin/env python

import rospy
import numpy as np
from bwrobot.msg import Arrays
import baxter_interface
import baxter_dataflow
from baxter_interface import settings
from teleop_utils.srv import GetTeleop, GetTeleopResponse
import sys
sys.path.insert(0, '/home/bpwilcox/catkin_ws/src/SOLAR_GP-ROS/scripts/')
from robot_controller import RobotController

class BaxterController(RobotController):
    """
    This class implements the RobotController base class for the Baxter Robot
    """
    def __init__(self, arm, joint_names, joint_topic):
        RobotController.__init__(self, joint_names, joint_topic)
        self.arm = arm
        # Set smoothness of angle filter
        self.coef = rospy.get_param('~filter_coef', 0.1)
        # Adds API support for controlling Baxter arm
        self.limb = baxter_interface.Limb(self.arm) 
        # Adds API support for controlling Baxter gripper
        self.gripper = baxter_interface.Gripper(self.arm)

    def jitter(self, njit, deg):

        Y = np.array(self.currentY).reshape(1,len(self.joint_names))   
        pert = deg * 0.0174533 * np.random.uniform(-1.,1.,(njit, np.size(Y,1)))
        YI = Y + pert

        for y in YI:        
            self.limb.move_to_joint_positions(dict(zip(self.joint_names,y.tolist())))
        return True
    
    def set_neutral(self):
        """
        Command Baxter to Neutral arm position
        """
        self.limb.move_to_neutral()
        self.nextY = self.currentY[:]
        return True

    def move_joints(self,timeout=15.0,
                                    threshold=settings.JOINT_ANGLE_TOLERANCE,
                                    test=None):
        """
        (Blocking) Commands the limb to the provided positions.

        Waits until the reported joint state matches that specified.

        This function uses a low-pass filter to smooth the movement.

        @type positions: dict({str:float})
        @param positions: joint_name:angle command
        @type timeout: float
        @param timeout: seconds to wait for move to finish [15]
        @type threshold: float
        @param threshold: position threshold in radians across each joint when
        move is considered successful [0.008726646]
        @param test: optional function returning True if motion must be aborted
        """

        def filtered_cmd():
            # First Order Filter - 0.2 Hz Cutoff
            cmd = self.joint_angles
            for idx, joint in enumerate(self.joint_names):
                # cmd[joint] = 0.012488 * self.nextY[idx] + 0.98751 * cmd[joint]
                cmd[joint] = self.coef * self.nextY[idx] + (1-self.coef) * cmd[joint]
            return cmd

        def genf(joint, angle):
            def joint_diff():
                return abs(angle - self.joint_angles[joint])
            return joint_diff

        diffs = [genf(j, a) for j, a in zip(self.joint_names, self.nextY)]

        self.limb.set_joint_positions(filtered_cmd())
        # baxter_dataflow.wait_for(
        #     test=lambda: callable(test) and test() == True or \
        #                 (all(diff() < threshold for diff in diffs)),
        #     timeout=timeout,
        #     timeout_msg=("%s limb failed to reach commanded joint positions"),
        #     rate=100,
        #     raise_on_error=False,
        #     body=lambda: self.limb.set_joint_positions(filtered_cmd())
        #     )

        return True

    def on_teleop(self, state):
        """
        Triggers jitter, neutral, or gripper open/close on teleoperator signal
        """
        if state.button6:
            self.jitter(5,3)
        if state.button7:
            self.set_neutral()
        if state.button4:
            self.gripper.close()
        elif state.button5:
            self.gripper.open()
        if state.button8:
            return False
        return True
    
def actuate():

    rospy.init_node('actuator_node')
    
    joint_names = rospy.get_param('~joints', ['s0', 's1', 'e1', 'w1'])
    joint_topic = rospy.get_param('~y_topic', 'robot/joint_states')
    arm = rospy.get_param('~arm', 'right')
    joint_names = [arm + '_' + joint for joint in joint_names]

    MoveRobot = BaxterController(arm, joint_names, joint_topic)
    MoveRobot.run()
        
if __name__ == '__main__':
    try:
        actuate()
    except rospy.ROSInterruptException:
        pass
