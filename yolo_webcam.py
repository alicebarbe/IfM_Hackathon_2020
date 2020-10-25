import cv2
import numpy as np
from yolo import YOLO, detect_video


def annotate_image(image, out_boxes, out_classes, out_scores):
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

        annotated_img = cv2.rectangle(annotated_img, (left, top),
                                      (right, bottom), (255, 0, 0), thickness)
        annotated_img = cv2.putText(annotated_img, label, text_origin,
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0))

    return annotated_img


def get_next_video_frame():
    res, frame = cam.read()
    if res:
        return frame
    else:
        return None

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

        for cls, box in zip(out_classes, out_boxes):
            if cls == "banana":
                top, left, bottom, right = box
                product = image[top:bottom, right:left]
                # run this through the dodginess detector

        out_image = annotate_image(image, out_boxes, out_classes, out_scores)

        cv2.imshow("Out", out_image)
        cv2.waitKey(delay)

