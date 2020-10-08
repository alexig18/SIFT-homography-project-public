#!/usr/bin/env python

from PyQt5 import QtCore, QtGui, QtWidgets
from python_qt_binding import loadUi

import cv2
import numpy as np
import sys

index_params = dict(algorithm=0, trees=5)
search_params = dict()
flann = cv2.FlannBasedMatcher(index_params, search_params)

class My_App(QtWidgets.QMainWindow):

    def __init__(self):
        super(My_App, self).__init__()
        loadUi("./SIFT_app.ui", self)

        self._cam_id = 0
        self._cam_fps = 10
        self._is_cam_enabled = False
        self._is_template_loaded = False

        self.browse_button.clicked.connect(self.SLOT_browse_button)
        self.toggle_cam_button.clicked.connect(self.SLOT_toggle_camera)

        self._camera_device = cv2.VideoCapture(self._cam_id)
        self._camera_device.set(3, 320)
        self._camera_device.set(4, 240)

        # Timer used to trigger the camera
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self.SLOT_query_camera)
        self._timer.setInterval(1000 / self._cam_fps)

    def SLOT_browse_button(self):
        dlg = QtWidgets.QFileDialog()
        dlg.setFileMode(QtWidgets.QFileDialog.ExistingFile)

        if dlg.exec_():
            self.template_path = dlg.selectedFiles()[0]

        pixmap = QtGui.QPixmap(self.template_path)
        self.template_label.setPixmap(pixmap)

        print("Loaded template image file: " + self.template_path)

    # Source: stackoverflow.com/questions/34232632/
    def convert_cv_to_pixmap(self, cv_img):
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        height, width, channel = cv_img.shape
        bytesPerLine = channel * width
        q_img = QtGui.QImage(cv_img.data, width, height, 
                    bytesPerLine, QtGui.QImage.Format_RGB888)
        return QtGui.QPixmap.fromImage(q_img)

    def SLOT_query_camera(self):
        ret, frame = self._camera_device.read()
        #TODO run SIFT on the captured frame
        #gray = cv2.imread(self.template_path)
        #source of algorithm: Sergio Canu’s tracking using Homography tutorial
        gray = cv2.cvtColor(cv2.imread(self.template_path),cv2.COLOR_BGR2GRAY)
        sift = cv2.xfeatures2d.SIFT_create()
        kp, des = sift.detectAndCompute(gray, None)

        #webcam frame
        gray_cam = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        kp_cam, des_cam = sift.detectAndCompute(gray_cam, None)

        matches = flann.knnMatch(des, des_cam, k=2)
        inliers = []
        #m = best matches, n = 2nd best matches
        for m,n in matches:
            if m.distance < 0.5 * n.distance:
                inliers.append(m)

        #img_kp = cv2.drawKeypoints(gray, kp, gray)
        #cv2.imshow("image", img_kp)
        #key=cv2.waitKey(1)

        matched_image = cv2.drawMatches(gray, kp, gray_cam, kp_cam, inliers, gray_cam)
        if len(inliers) > 8:
            original_pts = np.float32([kp[m.queryIdx].pt for m in inliers]).reshape(-1,1,2)
            train_pts = np.float32([kp_cam[m.trainIdx].pt for m in inliers]).reshape(-1,1,2)

            matrix, _ = cv2.findHomography(original_pts, train_pts, cv2.RANSAC, 5.0)

            #to have the image transform according to angle
            h, w = gray.shape
            outline = np.float32([[0,0],[0,h],[w,h],[w,0]]).reshape(-1,1,2)
            perspective = cv2.perspectiveTransform(outline, matrix)
            gray_cam = cv2.cvtColor(gray_cam, cv2.COLOR_GRAY2BGR)
            #connect lines=true, colour=blue, thickness of line=3
            homography = cv2.polylines(gray_cam, [np.int32(perspective)], True, (255, 0, 0), 3)
            
            pixmap = self.convert_cv_to_pixmap(homography)
        else:
            pixmap = self.convert_cv_to_pixmap(matched_image)


    
        self.live_image_label.setPixmap(pixmap)

    def SLOT_toggle_camera(self):
        if self._is_cam_enabled:
            self._timer.stop()
            self._is_cam_enabled = False
            self.toggle_cam_button.setText("&Enable camera")
        else:
            self._timer.start()
            self._is_cam_enabled = True
            self.toggle_cam_button.setText("&Disable camera")

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    myApp = My_App()
    myApp.show()
    sys.exit(app.exec_())
