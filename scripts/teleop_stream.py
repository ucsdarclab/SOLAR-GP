#!/usr/bin/env python

import rospy
import numpy as np
from teleop_utils import xbox_teleop, phantom_teleop
import sys
sys.path.insert(0, '/home/bpwilcox/catkin_ws/src/SOLAR_GP-ROS/scripts/utilities')
import trajectory_sender

def teleop():
    """
    This function creates the Teleoperator class which sends poses across topics both for 
    visualization and goal commands for Predictor. Support is parameterized from launch file
    for xbox teleoperation and teleoperation playback from a bag file
    """
    rospy.init_node('teleop_node')
    R = rospy.get_param('~teleop_pub_rate', 100)
    teleop_device = rospy.get_param('~device', 'bag')
    filename = rospy.get_param('~trajectory_filename', 'first_bag.bag')

    # Available Teleop Modes
    # devices = ['xbox', 'phantom', 'bag']
    if teleop_device == 'xbox':
        Teleoperator = xbox_teleop.XboxTel(R)
    elif teleop_device == 'bag':
        Teleoperator = trajectory_sender.TrajectorySender(filename, R)

    rate = rospy.Rate(R)

    while not rospy.is_shutdown():
        # Publish current teleoperator pose, next goal pose, and teleoperator goal path
        Teleoperator.pub_pose.publish(Teleoperator.currentPose)
        Teleoperator.pub_pose_next.publish(Teleoperator.nextPose)
        Teleoperator.pub_path.publish(Teleoperator.path)
        rate.sleep()

if __name__ == '__main__':
    try:
        teleop()
    except rospy.ROSInterruptException:
        pass
