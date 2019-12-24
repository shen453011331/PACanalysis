import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class PlotResult(object):
    def __init__(self):
        self.boolLocateshow = 1
        pass

    def plot_locate_result(self, image, points, wait_time=1):
        if self.boolLocateshow:
            width = 200
            height = 200
            coordinate = np.floor(points.min(0)-np.array([width/2, height/2]))
            coordinates = [int(coordinate[0]), int(coordinate[1])]
            if coordinates[0] >= 0 and coordinates[1] >= 0:
                y1 = int(coordinates[1])
                y2 = int(coordinates[1] + height)
                x1 = int(coordinates[0])
                x2 = int(coordinates[0] + width)
                cv2.namedWindow("locate_large",0)
                temp_image_large = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                cv2.rectangle(temp_image_large, (x1, y1), (x2, y2), (255, 0, 0), 5)
                cv2.imshow("locate_large", temp_image_large)
                cv2.waitKey(wait_time)
                image_slice = image[y1:y2, x1:x2]
                temp_points = points - coordinate
                cv2.namedWindow("locate", 0)
                temp_image = cv2.cvtColor(image_slice, cv2.COLOR_GRAY2BGR)
                cv2.circle(temp_image, (int(temp_points[0, 0]), int(temp_points[0, 1])), 5, (255, 255, 0), 1)
                cv2.circle(temp_image, (int(temp_points[1, 0]), int(temp_points[1, 1])), 5, (255, 255, 0), 1)
                cv2.imshow("locate", temp_image)
                cv2.waitKey(wait_time)

    def plot_locate_refine_result(self, image, points, points_l, waittime=1):
        if self.boolLocateshow:
            width = 100
            height = 100
            coordinate = np.floor(points.min(0) - np.array([width / 2, height / 2]))
            coordinates = [int(coordinate[0]), int(coordinate[1])]
            if coordinates[0] >= 0 and coordinates[1] >= 0:
                y1 = int(coordinates[1])
                y2 = int(coordinates[1] + height)
                x1 = int(coordinates[0])
                x2 = int(coordinates[0] + width)
                cv2.namedWindow("locate_large", 0)
                temp_image_large = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                cv2.rectangle(temp_image_large, (x1, y1), (x2, y2), (255, 0, 0), 5)
                cv2.imshow("locate_large", temp_image_large)
                cv2.waitKey(waittime)
                image_slice = image[y1:y2, x1:x2]
                temp_points = points - coordinate
                cv2.namedWindow("locate", 0)
                temp_image = cv2.cvtColor(image_slice, cv2.COLOR_GRAY2BGR)
                cv2.circle(temp_image, (int(temp_points[0, 0]), int(temp_points[0, 1])), 2, (255, 255, 0), 1)
                cv2.circle(temp_image, (int(temp_points[1, 0]), int(temp_points[1, 1])), 2, (255, 255, 0), 1)
                temp_points_l = points_l - coordinate
                cv2.circle(temp_image, (int(temp_points_l[0, 0]), int(temp_points_l[0, 1])), 2, (0, 0, 255), 1)
                cv2.circle(temp_image, (int(temp_points_l[1, 0]), int(temp_points_l[1, 1])), 2, (0, 0, 255), 1)
                cv2.imshow("locate", temp_image)
                cv2.waitKey(waittime)
                a = 0

    def plot_test_patch(self, new_patch, bbox_local_l, bbox_local_r):
        cv2.namedWindow('testPatch', 0)
        temp_new_patch = cv2.cvtColor(new_patch, cv2.COLOR_GRAY2BGR)
        y1 = int(bbox_local_l[1])
        y2 = int(bbox_local_l[1] + bbox_local_l[3])
        x1 = int(bbox_local_l[0])
        x2 = int(bbox_local_l[0] + bbox_local_l[2])
        cv2.rectangle(temp_new_patch, (x1, y1), (x2, y2), (255, 0, 0), 5)
        y1 = int(bbox_local_r[1])
        y2 = int(bbox_local_r[1] + bbox_local_r[3])
        x1 = int(bbox_local_r[0])
        x2 = int(bbox_local_r[0] + bbox_local_r[2])
        cv2.rectangle(temp_new_patch, (x1, y1), (x2, y2), (255, 0, 0), 5)
        cv2.imshow('testPatch', temp_new_patch)
        cv2.waitKey(0)



