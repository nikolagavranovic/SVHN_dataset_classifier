import os
import cv2
import numpy as np
import csv
from csv import reader


class Preprocessor():

    def preprocess_data(self, annot_path, new_annot_path, new_images_path, resize_dims, save_new_data = True):
        self.new_annot = {}  # datastructure for new annots, key is filename and value is label
        imgs, labels = [], []   # datastucture for images array
        img_cnt = 0

        with open(annot_path, 'r') as read_obj:
            csv_reader = reader(read_obj)
            header = next(csv_reader)
            # Check file as empty
            if header != None:
                # Iterate over each row after the header in the csv
                for row in csv_reader:
                
                    filename = row[0]
                    annots = [int(el) for el in row[1:]] # cast to int
                    try:
                        img = cv2.imread(filename)
                        img_height, img_width = img.shape[0], img.shape[1] # get image width and height

                        # if img_height > 80 and img_width > 40:  # this is the condition, but too many images have height < 80
                        # and is lost, so condition is reversed
                        if img_height > 40 and img_width > 80:
                            annots_inx = 0
                            if len(annots):  # if annots is not empty array
                                while annots_inx < len(annots):
                                    # getting box and label
                                    xbbox = annots[annots_inx]
                                    ybbox = annots[annots_inx + 1]
                                    width = annots[annots_inx + 2]
                                    height = annots[annots_inx + 3]
                                    label = annots[annots_inx + 4]
                                    # if bounding box is valid (does not exceed the image)
                                    if xbbox > 0 and ybbox > 0 and (xbbox + width) < img_width and (ybbox + height) < img_height:
                                        img_cnt += 1
                                        img_name = f"{img_cnt}.png"
                                        # taking area of image defined by bounding box (cropping)
                                        cropped_img = cv2.resize(img[ybbox:ybbox + height, xbbox:xbbox + width], resize_dims, interpolation = cv2.INTER_LINEAR)
                                        if save_new_data:
                                            cv2.imwrite(f"{new_images_path}/{img_name}", cropped_img)
                                        imgs.append(cropped_img)
                                        labels.append(label)
                                        self.new_annot[img_name] = label  # setting anotation

                                    annots_inx += 5 # index of anotation increments by 5 (bbox 4 elements and label 1)

                        print(f"File: {filename} processed")
                    except:
                        print(f"No file {filename}")
            
        self.imgs = np.array(imgs)
        self.labels = np.array(labels)
                   
        if save_new_data:
            # write labels for new images into csv
            with open(new_annot_path, 'w', newline='') as file:
                writer = csv.writer(file)
                for key, value in self.new_annot.items():
                    writer.writerow([key, str(value)])

        return self.imgs, self.labels

    def load_data(self, imgs_dir, annots_path):
        all_labels = {}
        with open(annots_path, "r") as f:
            reader = csv.reader(f)
            for line in reader:
                all_labels[line[0]] = int(line[1])

        imgs, labels = [], []
        for filename in os.listdir(imgs_dir):
            img = cv2.imread(os.path.join(imgs_dir, filename))
            if img is not None:
                imgs.append(img)
                labels.append(all_labels[filename])
            print(len(imgs))

        self.imgs, self.labels = np.array(imgs), np.array(labels) 

    def save_data_to_npfile(self, imgs_file_name, labels_file_name):
        with open(imgs_file_name, 'wb') as f:
            np.save(f, self.imgs)
        with open(labels_file_name, 'wb') as f:
            np.save(f, self.labels)   

    def load_from_array(self, file_path):
        with open(file_path, 'rb') as f:
            arr = np.load(f)
        return arr



    