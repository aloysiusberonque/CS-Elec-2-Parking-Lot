import cv2 as open_cv
import numpy as np
import logging
from drawing_utils import draw_contours
from colors import COLOR_GREEN, COLOR_WHITE, COLOR_BLUE

import cv2
import torch
import os
import numpy as np

import datetime


model = torch.hub.load('ultralytics/yolov5','yolov5n', pretrained=True)
# model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
print(model.names)

def class_to_label(x):
        """
        For a given label value, return corresponding string label.
        :param x: numeric label
        :return: corresponding string label
        """
        return model.names[int(x)]

def score_frame(frame):
        """
        Takes a single frame as input, and scores the frame using yolo5 model.
        :param frame: input frame in numpy/list/tuple format.
        :return: Labels and Coordinates of objects detected by model in the frame.
        """
        frame = [frame]
        results = model(frame)
        labels, cord = results.xyxyn[0][:, -1].numpy(), results.xyxyn[0][:, :-1].numpy()
        return labels, cord

def plot_boxes(results, frame):
        """
        Takes a frame and its results as input, and plots the bounding boxes and label on to the frame.
        :param results: contains labels and coordinates predicted by model on the given frame.
        :param frame: Frame which has been scored.
        :return: Frame with bounding boxes and labels ploted on it.
        """
        labels, cord = results
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        for i in range(n):
            row = cord[i]
            if row[4] >= 0.2:
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                centroid = (int((x1 + x2) / 2), int((y1 + y2) / 2 + (y2 - y1) / 4))
                bgr = (0, 255, 0)
                cv2.rectangle(frame, centroid, (centroid[0] + 1, centroid[1] + 1), (0, 255, 0), 1)
                cv2.putText(frame, model.names[int(labels[i])], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr, 1)

        return frame, centroid
    
cv2.destroyAllWindows()

class MotionDetector:
    LAPLACIAN = 1.4
    DETECT_DELAY = 1

    def __init__(self, video, coordinates, start_frame):
        self.video = video
        self.coordinates_data = coordinates
        self.start_frame = start_frame
        self.contours = []
        self.bounds = []
        self.mask = []
        self.status = []
        self.times = []

    def detect_motion(self):
        capture = open_cv.VideoCapture(self.video)
        capture.set(open_cv.CAP_PROP_POS_FRAMES, self.start_frame)

        coordinates_data = self.coordinates_data
        logging.debug("coordinates data: %s", coordinates_data)

        for p in coordinates_data:
            coordinates = self._coordinates(p)
            logging.debug("coordinates: %s", coordinates)

            rect = open_cv.boundingRect(coordinates)
            logging.debug("rect: %s", rect)

            new_coordinates = coordinates.copy()
            new_coordinates[:, 0] = coordinates[:, 0] - rect[0]
            new_coordinates[:, 1] = coordinates[:, 1] - rect[1]
            logging.debug("new_coordinates: %s", new_coordinates)

            self.contours.append(coordinates)
            self.bounds.append(rect)

            mask = open_cv.drawContours(
                np.zeros((rect[3], rect[2]), dtype=np.uint8),
                [new_coordinates],
                contourIdx=-1,
                color=255,
                thickness=-1,
                lineType=open_cv.LINE_8)

            mask = mask == 255
            self.mask.append(mask)
            logging.debug("mask: %s", self.mask)

        statuses = [True] * len(coordinates_data)
        times = [None] * len(coordinates_data)

        count = 0

        while capture.isOpened():
            ret, frame = capture.read()

            if ret == True:
                if count % 4 == 0:
                    results = score_frame(frame) # Score the Frame
                    labels, cord = results
                frame, centroid = plot_boxes(results, frame)

                # cv2.imshow('Video', frame)

                key = cv2.waitKey(20)
                if key == 27:
                    break
                count = count + 1
            else:
                break

            if frame is None:
                break

            if not ret:
                raise CaptureReadError("Error reading video capture on frame %s" % str(frame))

            new_frame = frame.copy()
            logging.debug("new_frame: %s", new_frame)

            position_in_seconds = capture.get(open_cv.CAP_PROP_POS_MSEC) / 1000.0

            for index, c in enumerate(coordinates_data):
                status = self.__apply(frame, labels, cord, coordinates_data, centroid)

                # When there has been no change in the parking spot status for a given amount of time
                # if times[index] is not None and self.same_status(statuses, index, status):
                #     print("No changes")
                #     times[index] = None
                #     continue

                # Where there is a change and the change is inside the time delay
                # if times[index] is not None and self.status_changed(statuses, index, status):
                #     if position_in_seconds - times[index] >= MotionDetector.DETECT_DELAY:
                #         print("There is a change with the time constraint")
                #         statuses[index] = status
                #         times[index] = None
                #     continue
                
                 # Where there is a change and the change is inside the time delay
                if times[index] is None and self.status_changed(statuses, index, status):
                    print("There is a change without the time constraint")
                    times[index] = position_in_seconds
                    statuses[index] = status
                    continue

            for i, s in enumerate(statuses):
                # print("statuses[index]:   ", s)
                # s[0] = False
                # s[1] = True
                # s[2] = False
                # s[3] = True
                for index, p in enumerate(coordinates_data):
                    coordinates = self._coordinates(p)
                    # print('coordinates ', coordinates)
                    if s[index]:
                        color = COLOR_BLUE
                        # print(f"Changed color to BLUE for coordinate {index} in status {s[index]}")
                    else:
                        color = COLOR_GREEN
                        # print(f"Changed color to GREEN for coordinate {index} in status {s[index]}")

                    draw_contours(new_frame, coordinates, str(p["id"] + 1), COLOR_WHITE, color)
                    
                open_cv.imshow(str(self.video), new_frame)

            k = open_cv.waitKey(1)
            if k == ord("q"):
                break
        capture.release()
        open_cv.destroyAllWindows()


    def __apply(self, frame, labels, cord, coordinates_data, centroid):
        # Initialize list of centroid coordinates of car instances inside the ROI
        centroids = []

        # Initialize list of parking space status
        status = [False] * len(coordinates_data)
        # print("Starting Status: ", status)
        for i in range(len(labels)):
            if len(cord[i]) == 5:

                # Check if the centroid coordinates of the bounding box are inside any of the ROIs
                for j, (roi, s) in enumerate(zip(self.mask, status)):
                    # if s[j] == True:
                    #     print('Occupied at index', j)
                    #     print("Current status ", status)
                    #     continue

                    y_idx = int(centroid[1] - self.bounds[j][1])
                    x_idx = int(centroid[0] - self.bounds[j][0])

                    y_idx = np.clip(y_idx, 0, roi.shape[0] - 1)
                    x_idx = np.clip(x_idx, 0, roi.shape[1] - 1)

                    if roi[y_idx, x_idx] == 1:
                        centroids.append((j, centroid))
                        status[j] = True
                        # print('Centroid inside at index ', j)
                        # print("Current status ", status)
                    else:
                        status[j] = False
                        # print('Centroid not inside at index', j)
                        # print("Current status ", status)

                    # if s[j] == True:
                    #     print('Occupied at index ', j)
                    #     print("Current status ", status)
                    print("Value of s[j]", s)

            else:
                print('Cord length is not 5')
            
        # Return the status of the detected parking space
        return status

    @staticmethod   
    def _coordinates(p):
        return np.array(p["coordinates"])

    @staticmethod
    def same_status(coordinates_status, index, status):
        return status == coordinates_status[index]

    @staticmethod
    def status_changed(coordinates_status, index, status):
        return status != coordinates_status[index]


class CaptureReadError(Exception):
    pass
