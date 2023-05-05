import cv2 as open_cv
import numpy as np
import logging
from drawing_utils import draw_contours
from colors import COLOR_GREEN, COLOR_WHITE, COLOR_RED

import cv2
import torch
import numpy as np

# model = torch.hub.load('ultralytics/yolov5','yolov5n', pretrained=True)
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')
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
        centroids = []
        for i in range(n):
            row = cord[i]
            if row[4] >= 0.2:
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                centroid = (int((x1 + x2) / 2), int((y1 + y2) / 2 + ((y2 - y1) / 8)*3))
                centroids.append(centroid)
                bgr = (0, 255, 0)
                cv2.rectangle(frame, centroid, (centroid[0] + 1, centroid[1] + 1), (0, 255, 0), 1)
                cv2.putText(frame, model.names[int(labels[i])], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr, 1)

        return frame, centroids
    
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
        capture_count = 0

        while capture.isOpened():
            
            ret, frame = capture.read()

            if ret == True:
                results = score_frame(frame)
                frame, centroids = plot_boxes(results, frame)
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
            
            status = self.__apply(coordinates_data, centroids, capture_count)
            
            for index, p in enumerate(coordinates_data):
                coordinates = self._coordinates(p)
                if status[index]:
                    color = COLOR_RED
                else:
                    color = COLOR_GREEN

                draw_contours(frame, coordinates, str(p["id"] + 1), COLOR_WHITE, color)
                    
            open_cv.imshow(str(self.video), frame)

            k = open_cv.waitKey(1)

            if k == ord("q"):
                break

            capture_count += 1
        capture.release()
        open_cv.destroyAllWindows()

    def __apply(self, coordinates_data, centroids, capture_count):
        from matplotlib.path import Path
        status = [False] * len(coordinates_data)
        for j, (roi, s) in enumerate(zip(self.mask, status)):
            coordinates = [tuple(coord) for coord in coordinates_data[j]['coordinates']]
            path = Path(coordinates)
            self_coordinates = self._coordinates(coordinates_data[j])
            for centroid in centroids:
                if path.contains_point(centroid):
                    status[j] = True

        # Return the status of frame with the labeled detected parking spaces
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
