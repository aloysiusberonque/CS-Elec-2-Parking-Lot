import cv2
import torch

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
        for i in range(n):
            row = cord[i]
            if row[4] >= 0.2:
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                bot_center = (int((x1 + x2) / 2), int((y1 + y2) / 2 + (y2 - y1) / 4))
                bgr = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 1)
                cv2.rectangle(frame, bot_center, (bot_center[0] + 1, bot_center[1] + 1), (0, 255, 0), 1)
                cv2.putText(frame, model.names[int(labels[i])], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr, 1)

        return frame
    
video = cv2.VideoCapture('parking_lot.mp4')
count = 0

while (video.isOpened()):
    ret, frame = video.read()

    if ret == True:
        if count % 4 == 0:
            results = score_frame(frame)
        frame = plot_boxes(results, frame) # Plot the boxes.

        cv2.imshow('Video', frame)

        key = cv2.waitKey(20)
        if key == 27:
            break
        count = count + 1
    else:
        break

video.release()
cv2.destroyAllWindows()