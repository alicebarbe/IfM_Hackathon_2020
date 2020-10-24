import os

images_path = "../roboflow/train/images"
labels_path = "../roboflow/train/labels"
imsize = 416


if __name__ == "__main__":
    for root, dirs, files in os.walk(images_path, topdown=False):
        print(root)
        for file in files:
            label_path = os.path.join(labels_path, file[:-4] + ".txt")
            labels=[]

            with open(label_path) as label_f:
                label_str = label_f.readline()
                cls, x_min, y_min, x_max, y_max = label_str.split(" ")
