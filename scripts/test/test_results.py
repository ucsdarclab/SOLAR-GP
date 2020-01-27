#!/usr/bin/env python

import rospy
import numpy as np
import rosbag
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped, Pose
from copy import deepcopy, copy
from baxter_core_msgs.msg import EndpointState
from bwrobot.msg import LocalGP, Errors, Results, Params, IntHeader, FloatHeader
from std_msgs.msg import Time, Header

class TestResults():
    """
    This class records and writes a custom Results msg to a rosbag. The results include the parameter configuration,
    update times of SOLAR_GP models, and the errors between goal pose and actual pose 
    """
    def __init__(self, test_topic, cur_topic, GP_topic, duration, params = Params(), savefile = 'test.bag'):
        """
        test_topic: name for end effector state topic
        cur_topic: name current goal pose/state topic 
        GP_topic: name for SOLAR_GP topic
        duration: period of time before grabbing next result
        params: Params message of current SOLGAR_GP param configuration 
        savefile: filename to save results
        """
        self.test_pose = Pose()
        self.actual_pose = Pose()
        self.results = Results()
        self.results.params = params
        self.duration = duration
        self.filename = savefile
        rospy.Subscriber(test_topic, EndpointState, self.test_callback, queue_size = 10)
        rospy.Subscriber(cur_topic, PoseStamped, self.cur_callback, queue_size = 10)
        rospy.Subscriber(GP_topic, LocalGP, self.GP_callback, queue_size = 10)
        self.pub_error = rospy.Publisher('error', FloatHeader, queue_size=10)
        self.i = 0
        self.j = 0
        self.running = False   
        rospy.Timer(rospy.Duration(self.duration), self.timer_callback)

    def run(self):
        self.clear()
        self.running = True

    def clear(self):
        self.results.errors = []
        self.results.updates = []
        self.i = 0
        self.j = 0

    def stop(self):
        self.running = False

    def save_data(self):
        try:
            bag = rosbag.Bag(self.filename, 'w')
            bag.write('results', self.results)
        finally:
            bag.close()
        # self.stop()


    def test_callback(self, msg):
        self.test_pose = msg.pose

    def cur_callback(self, msg):
        self.actual_pose = msg.pose
        
    def GP_callback(self, msg):
        update = IntHeader()
        update.header = Header(seq = self.i, stamp = rospy.Time.now())
        update.data = msg.M
        self.results.updates.append(update)

    def timer_callback(self, event):
        if self.running:
            error = FloatHeader()
            error.data = self.error_distance(self.actual_pose, self.test_pose)
            error.header = Header(seq = self.i, stamp = rospy.Time.now())
            self.results.errors.append(error)
            # self.pub_error.publish(error)
            self.i+=1
    
    def error_distance(self, pose1, pose2):
        """
        Computes squared error/distance between poses
        """
        x1 = np.array([pose1.position.x, pose1.position.y, pose1.position.z]).reshape(1,3)
        x2 = np.array([pose2.position.x, pose2.position.y, pose2.position.z]).reshape(1,3)
        error = np.dot(x2-x1, np.transpose(x2-x1))
        return error


# def collect_results():

#     rospy.init_node('results_node')
#     arm = rospy.get_param('~arm', 'right')
#     test_topic = 'robot/limb/' + arm + '/endpoint_state'
#     cur_topic = 'teleop_pose_next'
#     GP_topic = 'solarGP'
#     duration = 0.25
#     Results = TestResults(test_topic, cur_topic, GP_topic, duration)
#     rospy.spin()

# if __name__ == '__main__':
#     try:
#         collect_results()
#     except rospy.ROSInterruptException:
#         pass
