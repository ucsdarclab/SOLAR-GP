#!/usr/bin/env python

import rospy
import numpy as np
from geometry_msgs.msg import Point, PoseStamped
import GPy
import time
import sys
sys.path.insert(0, '/home/bpwilcox/catkin_ws/src/SOLAR_GP-ROS/scripts/core')
import osgpr_GPy
from SOLAR_core import LocalModels
from sensor_msgs.msg import JointState
from teleop_utils.srv import GetTeleop, GetTeleopResponse, SetPose
from teleop_utils import xbox_teleop
from std_msgs.msg import Float64
from bwrobot.srv import *
from bwrobot.msg import *
from copy import copy, deepcopy
from baxter_core_msgs.msg import EndpointState
    
import warnings
warnings.simplefilter('always', UserWarning)


class SolarPredictor():
    """
    This class creates the base Predictor module for SOLAR_GP. While running, requests are made from the 
    teleoperator service for the next goal pose and any relevant teleop commands. Predictions are published
    on a topic for the controller
    """
    def __init__(self, topic):
        """
        topic: name of end effector state topic
        """
        self.model = LocalModels()
        self.pred_pub = rospy.Publisher('prediction', Arrays, queue_size=10)
        self.curX = []
        self.curPose = PoseStamped()
        rospy.Subscriber('solarGP',LocalGP,self.model_callback)
        rospy.Subscriber(topic, EndpointState, self.x_callback, queue_size = 10)
        
        rospy.wait_for_service('set_teleop_pose')
        self.set_teleop_pose = rospy.ServiceProxy('set_teleop_pose', SetPose)
        
    def decode_ang(self,q):
        """
        Decode angles from sin/cos to radians
        """
        d = int(np.size(q,1)/2)
        decoding = np.arctan2(q[:,:d], q[:,d:]).reshape(np.size(q,0),d)
        return decoding

    def x_callback(self, msg):
        self.curPose.header = msg.header
        self.curPose.header.frame_id = '/teleop'
        self.curPose.pose = msg.pose
        self.curX = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]).reshape(1,3)

 
    def model_callback(self,LocMsg):
        """
        Unserializes SOLAR_GP model messages passed across ROS Topic into SOLAR_GP Objects
        """
        W = LocMsg.W
        W = np.diag([W[0],W[1],W[2]])

        M = LocMsg.M
        LocalData = []
        Models = []
        xdim = LocMsg.xdim
        ndim = LocMsg.ndim

        for L in LocMsg.localGPs:
            
            X_loc = []
            X_loc.append(L.numloc)
            X_loc.append(L.numloc)
            X_loc.append(np.array(L.xmean).reshape(1,xdim))
            X_loc.append(np.array(L.ymean).reshape(1,ndim))
            X_loc.append(True)
            
            LocalData.append(X_loc)
            X = np.empty([0,xdim]) 
            Y = np.empty([0,ndim]) 
    
            Z = np.empty([0,xdim]) 
            Z_old = np.empty([0,xdim])
            mu_old = np.empty([0,ndim])
            Su_old = np.empty([0,len(L.Z_old)])
            Kaa_old = np.empty([0,len(L.Z_old)])
    
            kern = GPy.kern.RBF(xdim,ARD=True)
            kern.variance = L.kern_var
            kern.lengthscale = L.kern_lengthscale
            
            for x,y in zip(L.X,L.Y):
                X = np.vstack((X,np.array(x.array).reshape(1,xdim)))       
                Y = np.vstack((Y,np.array(y.array).reshape(1,ndim)))       
                           
            for z in L.Z:
                Z = np.vstack((Z,np.array(z.array).reshape(1,xdim)))
                
            for z,mu,su,ka in zip(L.Z_old, L.mu_old,L.Su_old,L.Kaa_old):
                Z_old = np.vstack((Z_old,np.array(z.array).reshape(1,xdim)))        
                mu_old = np.vstack((mu_old,np.array(mu.array).reshape(1,ndim)))        
                Su_old = np.vstack((Su_old,np.array(su.array).reshape(1,len(L.Z_old))))        
                Kaa_old = np.vstack((Kaa_old,np.array(ka.array).reshape(1,len(L.Z_old))))    
                
            m = osgpr_GPy.OSGPR_VFE(X, Y, kern, mu_old, Su_old, Kaa_old, Z_old, Z)    
            m.kern.variance = L.kern_var
            m.kern.lengthscale = np.array(L.kern_lengthscale)
            m.likelihood.variance = L.likelihood_var
            Models.append(m)
            
        local = LocalModels()
        local.W = W
        local.M = M
        local.LocalData = LocalData
        local.Models = Models
        local.xdim = xdim
        local.ndim = ndim
        self.model = local 

    def on_teleop(self, state):
        print("On Teleop not implemented")
        return False
    
    def run(self):
        R = rospy.get_param('~predict_pub_rate', 100)
        rate = rospy.Rate(R)
        d = rospy.get_param('~max_distance', 1)
        Yexp = []
    
    
        rospy.wait_for_service('get_teleop')
        get_teleop = rospy.ServiceProxy('get_teleop', GetTeleop)
        teleop_state = GetTeleopResponse()
    
        rospy.wait_for_message('solarGP',LocalGP)
        
        while not rospy.is_shutdown():
            
            # Get teleoperation state
            teleop_state = get_teleop()
            
            # Check for teleop actions
            if not self.on_teleop(teleop_state):
                continue
            
            # Get next goal pose
            data = teleop_state.pose
            xnext = np.array([data.pose.position.x,data.pose.position.y,data.pose.position.z]).reshape(1,3)

            #  If the distance to the next goal point is larger than desired threshold, clip
            v = xnext - self.curX
            norm_v = np.linalg.norm(v)
            if norm_v > d:
                u = v/norm_v
                xnext = self.curX + d*u 

            # Predict output joint angles (sin/cos)
            try:
                Ypred, _ = self.model.prediction(xnext, Y_prev = Yexp)
                Yexp = Ypred
            except:
                warnings.warn("warning during prediction")
                pass
            if np.isnan(Yexp).any():
                warnings.warn("warning has prediction Nan")
                continue

            # Publish joint angles (radians) 
            Y = self.decode_ang(Yexp).astype(float) 
            self.pred_pub.publish(Y[0,:].tolist())
            rate.sleep()
        