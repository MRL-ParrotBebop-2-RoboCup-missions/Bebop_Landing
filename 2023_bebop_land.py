#!/usr/bin/env python


import cv2
import sys
import numpy as np
import time
import math
import cv2, cv_bridge
from nav_msgs.msg import Odometry
import rospy
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from std_msgs.msg import Empty
from pid import PID


class land():
    def __init__(self):
        
        self.lower_hsv = np.array([6, 121, 201]) 
        self.upper_hsv = np.array([23, 255, 255])

        self.Pyaw = 0.01
        self.Iyaw = 0.001
        self.Dyaw = 0
        self.bridge = cv_bridge.CvBridge()        
        self.twist = Twist()    
        self.fwd = Twist()   
              
        self.camera_pub = rospy.Publisher('/bebop/camera_control', Twist, queue_size=1)   
        self.pub_vel = rospy.Publisher('bebop/cmd_vel', Twist, queue_size=1)
        self.land_pub = rospy.Publisher('bebop/land', Empty, queue_size=10)   
        time.sleep(1)

        self.Proll = 0.00025
        self.Iroll = 0.0004
        self.Droll = 0.0

        self.Pyaw = 0.01
        self.Iyaw = 0.001
        self.Dyaw = 0

        self.pid_roll = PID(self.Proll, self.Droll, self.Iroll, -0.1, 0.1, -0.1, 0.1) 
        self.pid_yaw = PID(self.Pyaw, self.Dyaw, self.Iyaw, -0.5, 0.5, -0.1, 0.1)  
        self.pid_yaw = PID(self.Pyaw, self.Dyaw, self.Iyaw, -0.04, 0.04, -0.1, 0.1)
        self.turn_toggle = True
        self.camera_angle = -50
        self.camera_down_start_time = time.time()
        self.go_to_H_toggle = False
        self.autonomous_flag = False
        self.contour_counter = 0
        self.counter_drift = 0


    def camera_decrease(self):
        if  time.time() -  self.camera_down_start_time > 1:
            self.camera_angle = self.camera_angle - 3
            self.cam_down(self.camera_angle)
            self.camera_down_start_time = time.time()

    def callback(self, data):                  
        cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        
        self.H_detect(cv_image)

        cv2.imshow("Adjustment", cv_image )         
        k = cv2.waitKey(1)  
        if k == ord('o'):           
            self.autonomous_flag = not self.autonomous_flag        
        if k == 27:  # close on ESC key
            cv2.destroyAllWindows()
            rospy.signal_shutdown('interrupt')  
        
    def H_detect(self, cv_image):        
        cv_image_hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)     
        start_mask = cv2.inRange(cv_image_hsv, self.lower_hsv, self.upper_hsv)       
        start_mask = cv2.dilate(start_mask, np.ones((5, 5), np.uint8), iterations=9)
        start_mask = cv2.erode(start_mask, np.ones((5, 5), np.uint8), iterations=5)       
        start_mask = cv2.morphologyEx(start_mask, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8), iterations=1)
         
        contours_blk, _ = cv2.findContours(start_mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours_blk = list(contours_blk)       
        contours_blk.sort(key=cv2.minAreaRect)

        if self.turn_toggle:
            turn_twist = Twist()
            turn_twist.angular.z = 0.1 
            if self.autonomous_flag:
                self.pub_vel.publish(turn_twist)

        if len(contours_blk) > 0 :           
            self.was_line = 1            
            blackbox = cv2.minAreaRect(contours_blk[-1])                        
            setpoint = cv_image.shape[1] / 2 # w/2
            
            center = (int(blackbox[0][0]),int(blackbox[0][1]))              
            rollE = int(center[0] - setpoint)
            pitchE = cv_image.shape[0] - center[1] #h/2
                      
            self.contour_counter = self.contour_counter + 1
            # 0.5 w for turning left .................................
            if 0.5 * cv_image.shape[1] - center[0] < 20 and self.contour_counter > 10 and  self.go_to_H_toggle==False:      
                print('go to start toggle')          
                self.contour_counter = 0  
                self.go_to_H_toggle = True  
                self.turn_toggle = False           
            
            if self.go_to_H_toggle: 
                if 0.3 * cv_image.shape[0] - center[1] > 1:
                    print('counter_drift', self.counter_drift)
                    self.counter_drift = self.counter_drift +1   # a counter for first time seeing start sign and before zoom for canceling roll drift
                    print('counter_drift')
                    if self.counter_drift > 4: #low pass filter
                        self.counter_drift = 0
                        self.starter_drift_flag = True
                        print('                        self.starter_drift_flag  ', True )
                else:
                    self.starter_drift_flag = False
                    self.twist = Twist()
                            
                self.controller (yaw_error=0, roll_error=rollE)   
                self.camera_decrease()
               

                if self.camera_angle == -90 and pitchE<10:
                    msg_land = Empty()
                    self.land_pub.publish(msg_land)

            box = np.int0(cv2.boxPoints(blackbox))             
            cv2.drawContours(cv_image, [box], 0, (200, 200, 100), 3)
            # cv2.putText(cv_image, "yawE: " + str(00), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6, color=(0, 0, 0),thickness=2)
            # cv2.putText(cv_image, "rollE: " + str(rollE), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6, color=(0, 0, 0),thickness=2)
                       
    def controller(self, yaw_error, roll_error):

        self.pid_roll = PID(self.Proll, self.Droll, self.Iroll, -0.3, 0.3, -0.1, 0.1) 
        self.pid_yaw = PID(self.Pyaw, self.Dyaw, self.Iyaw, -0.15, 0.15, -0.1, 0.1) 
        
        yaw_treshold = 10
        roll_treshold = 75
        if self.starter_drift_flag:
            self.twist.linear.x = 0.09
              
        
        if abs(roll_error) <= roll_treshold and abs(yaw_error)<= yaw_treshold:            
            # if self.start_rope_toggle:
            self.twist.linear.x =  0.02 #sasan
            # self.twist.linear.x =  0.01 #mrl
            self.twist.linear.y = 0.0
            self.twist.angular.z = 0.0  
            # else:
            #     self.twist.linear.x =  0.09 #sasan
            #     # self.twist.linear.x =  0.07 #mrl
            #     self.twist.linear.y = 0.0
            #     self.twist.angular.z = 0.0         
            #     self.stateFlag = 'pitch'
                
        if abs(roll_error) > roll_treshold:
            # self.twist.linear.x = 0 #abs(self.pid_roll.update(roll_error)) * 0.9 for drift of roll ratio rate 
            self.twist.linear.y = - self.pid_roll.update(roll_error)
            self.twist.angular.z = 0.0                
            self.stateFlag = 'roll'
            
        if abs(yaw_error) > 5 and  abs(roll_error) < 200:
            self.twist.linear.x = 0 #0.004 
            self.twist.linear.y = 0.0
            self.twist.angular.z = -self.pid_yaw.update(yaw_error)                  
            self.stateFlag='yaw'

        if self.autonomous_flag:
            self.pub_vel.publish(self.twist)   
    
   

    def cam_down(self, angle):
        cam = Twist()
        cam.angular.y = angle 
        self.camera_pub.publish(cam)

if __name__ == '__main__':  
    rospy.init_node('takeoff_adjustment', anonymous=True)    
    
    la = land()   
    rospy.Subscriber('/bebop/image_raw', Image, la.callback)       
    time.sleep(3)            
         
    rospy.spin()

