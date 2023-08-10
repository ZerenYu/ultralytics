from __future__ import print_function
from logging.config import valid_ident
from ultralytics import YOLO
import os
import numpy as np
from sensor_msgs.msg import Image
import geometry_msgs.msg 
from cv_bridge import CvBridge, CvBridgeError
import rospy
import cv2
import time
# from vision_pickup.srv import yolo_mask, yolo_maskResponse


# model = YOLO("/home/cam/amazon_ws/runs/yolomodel/best.pt")
model = YOLO("/home/yuzeren/CAM/Amazon_ws/runs/segment/train11/weights/best.pt")
bridge = CvBridge()

def mask_generate(rgb_channel, mask_channel, depth_channel):
    rgb_msg = rospy.wait_for_message("/camera/color/image_raw", Image, timeout=None)
    rgb_data = bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
    depth_msg = rospy.wait_for_message("/camera/aligned_depth_to_color/image_raw", Image, timeout=None)
    # depth_data = bridge.imgmsg_to_cv2(depth_msg, "16UC1")
    rgb_data = np.asarray(rgb_data)
    # depth_data = np.asarray(depth_data)
    timestamp = int(time.time()*1000)
    result = model(rgb_data)
    mask = np.ones((480, 640))
    if result[0].masks:
        count = 0 
        segs = np.asarray(result[0].masks.segments)
        for seg in segs:
            seg = np.asarray(seg)
            seg = list(seg.reshape(seg.shape[0] * seg.shape[1]))
            seg.insert(0, 0)
            count += 1
            processed = np.asarray([[[int(seg[i+1]*640), int(seg[i+2]*480)] for i in range(0, len(seg)-1, 2)]])
            if(processed.shape[1] >= 3):
                cv2.fillPoly(mask, pts=processed, color=0)

            break
        rospy.loginfo("Found {} packages".format(count))
    else:
        rospy.loginfo("No package found")
    rgb_data = cv2.cvtColor(rgb_data, cv2.COLOR_BGR2RGB)
    rgb_data[:,:,1] = rgb_data[:,:,1] * mask
    mask = mask.astype(np.uint8)
    m_img = Image()
    m_img.header.stamp = rospy.Time.from_sec(int(timestamp/1000000))
    m_img.height = 480
    m_img.width = 640
    m_img.step = 3 * 640 * 480
    m_img.encoding = 'rgb8'
    m_img.data = rgb_data.flatten().tolist()
    rgb_channel.publish(m_img)

    mask = mask.astype(np.uint8)
    mask[mask==1] = 2
    mask[mask==0] = 255
    mask[mask==2] = 0
    m_img = Image()
    m_img.header.stamp = rospy.Time.from_sec(int(timestamp/1000000))
    m_img.height = 480
    m_img.width = 640
    m_img.step = 3 * 640 * 480
    m_img.encoding = 'rgb8'
    m_img.data = np.dstack((mask, mask, mask)).flatten().tolist()
    mask_channel.publish(m_img)

    depth_msg.header.stamp = rospy.Time.from_sec(int(timestamp/1000000))
    depth_channel.publish(depth_msg)

    return


def srv_manager():
    rgb = rospy.Publisher('pickup/rgb', Image, queue_size=4)
    mask = rospy.Publisher('pickup/mask', Image, queue_size=4)
    depth = rospy.Publisher('pickup/depth', Image, queue_size=4)
    rospy.init_node('pickup', anonymous=True)
    rate = rospy.Rate(10) # 10hz
    rospy.loginfo("yolo node on")
    while not rospy.is_shutdown():
        mask_generate(rgb, mask, depth)
        rate.sleep()



if __name__ == "__main__":
    srv_manager()