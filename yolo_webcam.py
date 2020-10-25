import cv2
import numpy as np
from yolo import YOLO, detect_video
from enum import Enum

def annotate_image(image, out_boxes, out_classes, out_scores, out_quality):
    class_names = yolo._get_class()
    thickness = (image.shape[0] + image.shape[1]) // 300

    annotated_img = image.copy()
    for i, c in reversed(list(enumerate(out_classes))):
        predicted_class = class_names[c]
        box = out_boxes[i]
        score = out_scores[i]

        label = '{} {:.2f}'.format(predicted_class, score)

        top, left, bottom, right = box
        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(image.shape[1], np.floor(bottom + 0.5).astype('int32'))
        right = min(image.shape[0], np.floor(right + 0.5).astype('int32'))
        print(label, (left, top), (right, bottom))

        text_origin = (left, top + 1)

        if out_quality[i] == ProductState.GOOD:
            colour = (0,255,0)
        elif out_quality[i] == ProductState.BAD:
            colour = (0, 0, 255)
        elif out_quality[i] == ProductState.INCORRECT_ITEM:
            colour = (255, 0, 0)


        annotated_img = cv2.rectangle(annotated_img, (left, top),
                                      (right, bottom), colour, thickness)
        annotated_img = cv2.putText(annotated_img, label, text_origin,
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, colour)

    return annotated_img

def get_item_states(out_classes, out_boxes, image, expected_class):
    out_quality = []
    for cls, box in zip(out_classes, out_boxes):
        if cls == expected_class:
            top, left, bottom, right = box
            product = image[top:bottom, right:left]
            out_quality.append(ProductState.GOOD)  # run actual detector here
        else:
            out_quality.append(ProductState.INCORRECT_ITEM)

    return out_quality

def get_next_video_frame():
    res, frame = cam.read()
    if res:
        return frame
    else:
        return None

class ProductState(Enum):
    GOOD = 0
    BAD = 1
    INCORRECT_ITEM = 2

if __name__ == "__main__":
    FRAMERATE = 30
    VIDEO_FILE = None
    delay = int(1000 / FRAMERATE)

    yolo = YOLO()

    if VIDEO_FILE is None:
        cam = cv2.VideoCapture(0)
    else:
        cam = cv2.VideoCapture(VIDEO_FILE)
    cv2.namedWindow("Out")

    while True:
        f = get_next_video_frame()
        if f is None:
            print("Error getting frame")
            continue

        image = f
        # out_img = letterbox_image_cv(image, (416, 416))

        out_boxes, out_scores, out_classes = yolo.detect_bounding_boxes(image)
        out_quality = get_item_states(out_classes, out_boxes, image, "apple")

        out_image = annotate_image(image, out_boxes, out_classes, out_scores, out_quality)

        cv2.imshow("Out", out_image)
        cv2.waitKey(delay)

