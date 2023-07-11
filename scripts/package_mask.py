from __future__ import print_function
from logging.config import valid_ident
from ultralytics import YOLO
import os
import numpy as np
import rospy
import cv2
from vision_pickup.srv import yolo_mask, yolo_maskResponse


model = YOLO("/home/cam/amazon_ws/runs/yolomodel/best.pt")

def mask_generate(req):
    rospy.loginfo("HEARD {} {}".format(req.timestamp, req.store_dir))
    timestamp = req.timestamp
    store_dir = req.store_dir
    result = model(os.path.join(store_dir, "rgb", timestamp+'.png'))
    if result[0].masks:
        count = 0 
        img = np.zeros((480, 640))
        with open(os.path.join(store_dir, 'mask_pose', timestamp+".txt"), "w+") as fp:
            segs = np.asarray(result[0].masks.segments)
            for seg in segs:
                seg = np.asarray(seg)
                seg = list(seg.reshape(seg.shape[0] * seg.shape[1]))
                seg.insert(0, 0)
                line = ' '.join(str(num) for num in seg) + '\n'
                fp.write(line)   
                count += 1
                processed = np.asarray([[[int(seg[i+1]*640), int(seg[i+2]*480)] for i in range(0, len(seg)-1, 2)]])
                cv2.fillPoly(img, pts=processed, color=255)
                break
        cv2.imwrite(os.path.join(store_dir, "mask_img", timestamp+'.png'), img)
        rospy.loginfo("Found {} packages".format(count))
        rospy.loginfo("Write to {}".format(timestamp))

        return yolo_maskResponse(count)
    else:
        return yolo_maskResponse(0)


def srv_manager():
    rospy.init_node('yolo_mask')
    s = rospy.Service('gen_mask', yolo_mask, mask_generate)
    rospy.loginfo("yolo mask started")
    rospy.spin()



if __name__ == "__main__":
    srv_manager()