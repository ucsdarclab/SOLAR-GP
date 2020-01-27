#!/usr/bin/env python

import rospy
from nav_msgs.msg import Path
from bwrobot.srv import SaveToFile
from copy import copy, deepcopy
import rosbag


class TrajectorySaver():
    """
    This class saves trajectories from a Path topic to a rosbag
    """
    def __init__(self, path_topic):
        rospy.Subscriber(path_topic, Path, self.path_callback, queue_size = 10)
        self.currentPath = Path()
        # Exposes service to save a current Path
        self.service= rospy.Service('save_trajectory',SaveToFile, self.save_callback)
    def path_callback(self, msg):
        self.currentPath = msg

    def save_callback(self, req):
        try:
            bag = rosbag.Bag(req.filename, 'w')
            bag.write('trajectory', self.currentPath)
        finally:
            bag.close()
            return True
        return False

def saver():

    rospy.init_node('trajectory_saver_node')
    path_topic = rospy.get_param('~path_topic', 'teleop_path')
    TrajectorySaver(path_topic)
    rospy.spin()

if __name__ == '__main__':
    try:
        saver()
    except rospy.ROSInterruptException:
        pass