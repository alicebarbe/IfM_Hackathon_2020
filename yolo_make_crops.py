import argparse
from yolo import YOLO, detect_video
from PIL import Image
import numpy as np
import os

def make_individual_product_database(yolo, image_paths):
    try:
        os.mkdir("dataset")
    except OSError as error:
        print(error)

    class_count = {}
    class_names = yolo._get_class()

    for classname in class_names:
        class_count[classname] = 0
        try:
            os.mkdir(os.path.join("dataset", classname))
        except OSError as error:
            print(error)



    for img in image_paths:
        try:
            image = Image.open(img)
        except:
            print('Open Error! Try again!')
            continue
        else:
            out_boxes, out_scores, out_classes = yolo.detect_bounding_boxes(image)

            for i, c in reversed(list(enumerate(out_classes))):
                predicted_class = class_names[c]
                class_count[predicted_class] += 1
                box = out_boxes[i]
                score = out_scores[i]

                top, left, bottom, right = box

                top = max(0, np.floor(top + 0.5).astype('int32'))
                left = max(0, np.floor(left + 0.5).astype('int32'))
                bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
                right = min(image.size[0], np.floor(right + 0.5).astype('int32'))

                area = (left, top, right, bottom)
                cropped_img = image.crop(area)
                cropped_img.save("dataset/{}/{}".format(predicted_class, str(class_count[predicted_class]) + ".jpg"))

    yolo.close_session()

if __name__ == "__main__":

    train_folder
    train_image_paths =
    make_individual_product_database(YOLO())