#!/usr/bin/env python

import rospy
from nav_msgs.msg import Path
from copy import copy, deepcopy
import rosbag
from geometry_msgs.msg import PoseStamped
from teleop_utils.srv import SetPose, GetTeleop, GetTeleopResponse, GetTeleopRequest
from std_msgs.msg import Header, Bool
from std_srvs.srv import Empty, EmptyResponse

class TrajectorySender():
    """
    This class mimics a Teleoperator interface and updates current goal poses
    from a trajactory playback rosbag on a timer
    """
    def __init__(self, filename, rate):
        """
        filename: name of bag file for trajectory. Contains a Path
        rate: rate at which to update goal pose
        """

        self.filename = filename
        self.button1 = False
        self.button2 = True
        self.button3 = False
        self.button4 = False
        self.button5 = False
        self.button6 = False
        self.button7 = False
        self.button8 = False

        self.nextPose = PoseStamped()
        self.currentPose = PoseStamped()
        self.path = Path()
        self.path.header.stamp = rospy.Time.now()        
        self.path.header.frame_id = '/teleop'
        self.i = 0
        self._on_init()   
        self.pub_pose = rospy.Publisher('teleop_pose', PoseStamped, queue_size=10)
        self.pub_path = rospy.Publisher('teleop_path', Path, queue_size=10)
        self.pub_pose_next = rospy.Publisher('teleop_pose_next', PoseStamped, queue_size=10)        
        self.setPose_service = rospy.Service('set_teleop_pose',SetPose, self.set_pose)
        self.getTeleop_service = rospy.Service('get_teleop',GetTeleop, self.get_teleop)
        self.restart_service = rospy.Service('restart_bag', Empty, self.restart)
        self.run_service = rospy.Service('run_bag', Empty, self.run)
        self.stop_service = rospy.Service('stop_bag', Empty, self.stop)
        self.running = False

        self.pub_done = rospy.Publisher('trajectory_finished', Bool, queue_size=10, latch=True)

        # self._on_init()
        self.duration = 1.0/rate
        rospy.Timer(rospy.Duration(self.duration), self.timer_callback)
    
    def _on_init(self):
        bag = rosbag.Bag(self.filename)
        for msg in bag.read_messages(topics=['trajectory']):
            self.path = msg.message
        bag.close()
        self.path.header.stamp = rospy.Time.now()
        self.path.header.frame_id = '/teleop'
        self.nextPose = self.path.poses[self.i]
        self.nextPose.header.stamp = rospy.Time.now()
        self.nextPose.header.frame_id = '/teleop'
        self.currentPose = self.nextPose

    def timer_callback(self, event):
        if self.running:
            self.path.header.stamp = rospy.Time.now()
            if self.i < len(self.path.poses):
                self.nextPose = self.path.poses[self.i]
                self.nextPose.header.stamp = rospy.Time.now()
                self.nextPose.header.frame_id = '/teleop'
                self.currentPose = self.nextPose
                self.i+=1
            else:
                self.button8 = True
            self.pub_done.publish(self.button8)

    def set_pose(self,req):
        self.currentPose = req.pose
        return True
        
    def get_teleop(self,req):
        return self.nextPose, self.button1, self.button2, self.button3, self.button4, self.button5, self.button6, self.button7, self.button8

    def restart(self, req):
        """
        Restarts trajectory path from the beginning
        """
        self.i = 0
        # self.button8 = False
        return EmptyResponse()

    def run(self, req):
        self.running = True
        self.button8 = False
        return EmptyResponse()
        
    def stop(self, req):
        self.running = False
        self.button8 = True
        return EmptyResponse()

# def sender():

#     rospy.init_node('trajectory_sender_node')
#     filename = rospy.get_param('~trajectory_filename', 'first_bag.bag')
#     R = rospy.get_param('~playback_rate', 10)
#     rate = rospy.Rate(R)

#     TrajBag = TrajectorySender(filename, R)

#     while not rospy.is_shutdown():
#         TrajBag.pub_pose.publish(TrajBag.nextPose)
#         TrajBag.pub_path.publish(TrajBag.path)
#         # TrajBag.get_teleop(GetTeleopRequest())
#         rate.sleep()


# if __name__ == '__main__':
#     try:
#         sender()
#     except rospy.ROSInterruptException:
#         pass