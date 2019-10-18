#!/usr/bin/env python

import rospy
import numpy as np
from sensor_msgs.msg import Image
import sys

OPENCV_PATH = "/home/takeuchi/.local/lib/python2.7/site-packages"
sys.path = [OPENCV_PATH] + sys.path
from cv_bridge import CvBridge
import cv2


class Dl_edge_detector():
    def __init__(self):
        self.INPUT_IMAGE = rospy.get_param(
            '~input_image', "/head_mount_kinect/hd/image_color_rect_desktop")
        self.prototxt = rospy.get_param(
            '~prototxt', "deploy.prototxt")
        self.caffemodel = rospy.get_param(
            '~caffemodel', "hed_pretrained_bsds.caffemodel")
        self.resize = rospy.get_param(
            '~resize', True)
        self.width = rospy.get_param(
            '~width', 1000)
        self.height = rospy.get_param(
            '~height', 1000)
        self.net = cv2.dnn.readNet(self.prototxt, self.caffemodel)
        cv2.dnn_registerLayer('Crop', CropLayer)
        self.bridge = CvBridge()
        self.pub = rospy.Publisher("/dl_detected_edge", Image, queue_size=1)
        self.subscribe()

    def subscribe(self):
        self.image_sub = rospy.Subscriber(self.INPUT_IMAGE,
                                          Image,
                                          self.callback)

    def callback(self, msg):
        rospy.loginfo("dl_edge_detction")
        img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        if(self.resize):
            size = (self.width, self.height)
        else:
            size = (img.shape[0], img.shape[1])
        inp = cv2.dnn.blobFromImage(img, scalefactor=1.0,
                                    size=size,
                                    mean=(104.00698793,
                                          116.66876762,
                                          122.67891434),
                                    swapRB=False, crop=False)
        self.net.setInput(inp)
        out = self.net.forward()
        out = out[0, 0]
        out = cv2.resize(out, (img.shape[1], img.shape[0]))
        out = 255 * out
        out = out.astype(np.uint8)
        msg_out = self.bridge.cv2_to_imgmsg(out, "mono8")
        msg_out.header = msg.header
        self.pub.publish(msg_out)


class CropLayer(object):
    def __init__(self, params, blobs):
        self.xstart = 0
        self.xend = 0
        self.ystart = 0
        self.yend = 0

    def getMemoryShapes(self, inputs):
        inputShape, targetShape = inputs[0], inputs[1]
        batchSize, numChannels = inputShape[0], inputShape[1]
        height, width = targetShape[2], targetShape[3]

        self.ystart = (inputShape[2] - targetShape[2]) // 2
        self.xstart = (inputShape[3] - targetShape[3]) // 2
        self.yend = self.ystart + height
        self.xend = self.xstart + width

        return [[batchSize, numChannels, height, width]]

    def forward(self, inputs):
        return [inputs[0][:, :, self.ystart:self.yend, self.xstart:self.xend]]


def main(args):
    rospy.init_node("dl_edge_detector", anonymous=False)
    dl_edge_detector = Dl_edge_detector()
    rospy.spin()


if __name__ == '__main__':
    main(sys.argv)
