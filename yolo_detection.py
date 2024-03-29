import cv2
import numpy as np

CONFIG = './car_detection/yolov3.cfg'
WEIGHTS = './car_detection/yolov3.weights'
CLASSES = './car_detection/yolov3.txt'


scale = 0.00392
COLORS = {'car':(255,255,0), 'bicycle':(0,255,0),'truck':(0,0,255),'bus':(0,255,255),'motorbike':(255,0,0), 'person':(128,255,0)}
net = cv2.dnn.readNet(WEIGHTS, CONFIG)
classes = None
with open(CLASSES, 'r') as f:
    classes = [line.strip() for line in f.readlines()]


def get_output_layers(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers


def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = str(classes[class_id])
    if(label in ('car','bicycle','truck','bus','motorbike','person')):
        color = COLORS[label]
        cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)
        # print(x,y,x_plus_w,y_plus_h)
        # cv2.rectangle(img, (x,y), (x_plus_w,y-21), color, -1)
        cv2.putText(img, label,(x,y_plus_h+15),cv2.FONT_HERSHEY_SIMPLEX,0.7, (255,255,255),1,cv2.LINE_AA)

def detect_obj(image):
    # image = cv2.imread(IMAGE_PATH)
    Width = image.shape[1]
    Height = image.shape[0]

    # COLORS = np.random.uniform(0, 255, size=(len(classes), 3))


    blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)

    net.setInput(blob)

    outs = net.forward(get_output_layers(net))

    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4


    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])


    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    ret_arr=[]
    for i in indices:
        i = i[0]
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))
        label = str(classes[class_id])
        if(label in ('car','bicycle','truck','bus','motorbike','person')):
            ret_arr.append([[round(x), round(y+h)], [round(x+w), round(y)]])


    return image, ret_arr
    # cv2.imshow("object detection", image)
    # cv2.waitKey()
    #
    # cv2.imwrite("object-detection.jpg", image)
    # cv2.destroyAllWindows()
