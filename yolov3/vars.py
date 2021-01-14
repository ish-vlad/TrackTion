class Yolo:
    LABELS_FILE = 'darknet/data/coco.names'
    CONFIG_FILE = 'darknet/cfg/yolov3.cfg'
    WEIGHTS_FILE = 'darknet/yolov3.weights.txt'
    CONFIDENCE_THRESHOLD = 0.75

class Image:
    IMG_SHAPE_X = 384
    IMG_SHAPE_Y = 288