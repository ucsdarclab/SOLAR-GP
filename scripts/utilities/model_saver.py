#!/usr/bin/env python

import rospy
from bwrobot.msg import LocalGP
from bwrobot.srv import SaveToFile
from copy import copy, deepcopy
import rosbag


class ModelSaver():
    """
    This clas saves a SOLAR_GP model custom ROS topic msg to a rosbag
    """
    def __init__(self, model_topic):
        rospy.Subscriber(model_topic, LocalGP, self.model_callback, queue_size = 10)
        self.model = LocalGP
        self.service= rospy.Service('save_model',SaveToFile, self.save_callback)
    
    def model_callback(self, msg):
        self.model = msg

    def save_callback(self, req):
        try:
            bag = rosbag.Bag(req.filename, 'w')
            bag.write('model', self.model)
        finally:
            bag.close()
            return True
        return False

def saver():

    rospy.init_node('model_saver_node')
    model_topic = rospy.get_param('~model_topic', 'localGP')
    ModelSaver(model_topic)
    rospy.spin()



if __name__ == '__main__':
    try:
        saver()
    except rospy.ROSInterruptException:
        pass