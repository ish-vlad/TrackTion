import numpy as np
import time
import cv2
import imutils
from imutils.video import FPS
from imutils.video import VideoStream
from yolov3.vars import Yolo, Image
import pandas as pd
import os
from mall.mapping import get_coords

LABELS_FILE = Yolo.LABELS_FILE
CONFIG_FILE = Yolo.CONFIG_FILE
WEIGHTS_FILE = Yolo.WEIGHTS_FILE
CONFIDENCE_THRESHOLD = Yolo.CONFIDENCE_THRESHOLD

IMG_SHAPE_X = Image.IMG_SHAPE_X
IMG_SHAPE_Y = Image.IMG_SHAPE_Y

def object_tracking_table(INPUT_FILE, OUTPUT_FILE, show_output_image=False, save_csv=True):
    H = None
    W = None

    rows = []

    fps = FPS().start()

    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(OUTPUT_FILE, fourcc, 30,
                             (IMG_SHAPE_X, IMG_SHAPE_Y), True)

    LABELS = open(LABELS_FILE).read().strip().split("\n")

    np.random.seed(4)
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
                               dtype="uint8")

    net = cv2.dnn.readNetFromDarknet(CONFIG_FILE, WEIGHTS_FILE)

    vs = cv2.VideoCapture(INPUT_FILE)

    # determine only the *output* layer names that we need from YOLO
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    cnt = -1;
    while True:
        cnt += 1
        if not cnt % 5:
            # print('\n')
            print("Frame number", cnt)
        try:
            (grabbed, image) = vs.read()
        except:
            break
        try:
            blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
                                     swapRB=True, crop=False)
        except:
            break

        net.setInput(blob)
        if W is None or H is None:
            (H, W) = image.shape[:2]
        layerOutputs = net.forward(ln)

        # initialize our lists of detected bounding boxes, confidences, and
        # class IDs, respectively
        boxes = []
        confidences = []
        classIDs = []

        # loop over each of the layer outputs
        for output in layerOutputs:
            # loop over each of the detections
            for detection in output:
                # extract the class ID and confidence (i.e., probability) of
                # the current object detection
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                # filter out weak predictions by ensuring the detected
                # probability is greater than the minimum probability
                if confidence > CONFIDENCE_THRESHOLD:
                    # scale the bounding box coordinates back relative to the
                    # size of the image, keeping in mind that YOLO actually
                    # returns the center (x, y)-coordinates of the bounding
                    # box followed by the boxes' width and height
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")

                    # use the center (x, y)-coordinates to derive the top and
                    # and left corner of the bounding box
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    # update our list of bounding box coordinates, confidences,
                    # and class IDs
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        # apply non-maxima suppression to suppress weak, overlapping bounding
        # boxes
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD,
                                CONFIDENCE_THRESHOLD)

        # ensure at least one detection exists
        if len(idxs) > 0:
            # loop over the indexes we are keeping
            for i in idxs.flatten():
                # extract the bounding box coordinates
                # print(i)
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])

                x_lower_middle = x + w / 2
                y_lower_middle = y + h
                x_cm, y_cm = get_coords(x_lower_middle, y_lower_middle)

                # print(f'x: {x}, y: {y}')
                # print(f'width: {w}, height: {h}')

                row = [
                    i,  # id
                    cnt,  # frame_num
                    h,  # box_h
                    w,  # box_w
                    x,  # box_xc
                    y,  # box_yc
                    np.round(x_cm, 4), # box_xc_cm
                    np.round(y_cm, 4) # box_yc_cm
                ]

                rows.append(row)

                color = [int(c) for c in COLORS[classIDs[i]]]

                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
                cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, color, 2)

        # show the output image
        if show_output_image:
            cv2.imshow("output", cv2.resize(image,(800, 600)))
        writer.write(cv2.resize(image, (IMG_SHAPE_X, IMG_SHAPE_Y)))
        fps.update()
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    fps.stop()

    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

    # do a bit of cleanup
    cv2.destroyAllWindows()

    # release the file pointers
    print("[INFO] cleaning up...")
    writer.release()
    vs.release()

    __table_to_csv(INPUT_FILE, rows, save_csv)

def __table_to_csv(INPUT_FILE, rows, save_csv=True):
    arr = np.vstack(rows)
    filename = INPUT_FILE.split('.')[0]
    ds_name = np.repeat(filename, arr.shape[0])[:, None]
    matrix = np.hstack((ds_name, arr))

    columns = ['ds_name', 'id', 'frame_num', 'box_height', 'box_width',
               'box_xc_pixels', 'box_yc_pixels', 'box_xc_cm', 'box_yc_cm']

    df = pd.DataFrame(matrix, columns=columns)
    df = df.apply(pd.to_numeric, errors='ignore')

    if save_csv:
        path_save = os.path.join('csv-files', f'{filename}.csv')
        df.to_csv(path_save, index=False)

    return df