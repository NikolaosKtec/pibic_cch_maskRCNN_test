# %%
# import basic libraries
import os
from os import listdir
from random import shuffle
import sys
import json
import datetime
import math

# ignore warnings
import warnings
warnings.filterwarnings("ignore")

# import advance libraries
from xml.etree import ElementTree
import skimage.draw
import cv2
import imgaug

# %%
# ROOT
ROOT = os.getcwd()
MODEL = os.path.join(ROOT,'model')
DATASET = os.path.join(ROOT,'dataset')
# TRAIN = os.path.join(DATASET,'train')
# VALIDATION = os.path.join(DATASET,'validation')
# anotation and image source
IMAGE_DIR =  os.path.join(DATASET,'images')
ANNOTATIONS_DIR = os.path.join(DATASET,'annotations')

# %%
# !alias pip=/usr/bin/pip

# %%
# import mask rcnn libraries
from mrcnn.utils import Dataset
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from mrcnn.visualize import display_instances
from mrcnn.utils import extract_bboxes
from mrcnn.utils import compute_ap
from mrcnn.model import load_image_gt
from mrcnn.model import mold_image
from mrcnn import visualize

# %% [markdown]
# <h1>change dir and download model</h1>
# <p>use:</p>
# <span>!wget https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5</span>

# %%
# import matplotlib library
import matplotlib.pyplot as plt

# import numpy libraries
import numpy as np
from numpy import zeros
from numpy import asarray
from numpy import expand_dims
from numpy import mean

# import keras libraries
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

# inline matplotlib
%matplotlib inline


# %%

class CchDataset(Dataset):
    #  prepare dataset and ignore images without annotations
    def prepare_dataset(self,IMAGE_DIR, ANNOTATIONS_DIR):

        images = []
        annotations = []
        image_id_list = []
        images_dir_list = listdir(IMAGE_DIR)
        shuffle(images_dir_list)
        
        for filename in images_dir_list:
            
            # extract image id
            image_id = filename[:-4] # used to skip last 4 chars which is '.jpg' (class_id.jpg)
            
            
            img_path = IMAGE_DIR +'/'+filename
            ann_path = ANNOTATIONS_DIR+'/'+ image_id + '.xml'
            
            if os.path.exists(img_path) and os.path.exists(ann_path):

                images.append(img_path)
                annotations.append(ann_path)
                image_id_list.append(image_id)

            # print(f'exist? >> {os.path.exists(img_path)} >> path_ing: {img_path}')
            # print(f'exist? >> {os.path.exists(ann_path)} >> path_ann: {ann_path}')
            
        return (images, annotations,image_id_list)

            
    
    # load_dataset function is used to load the train and test dataset
    def load_dataset(self,IMAGES_LIST, ANNOTATIONS_LIST, IMAGE_ID_LIST, is_train=True):
        
        # we add a class that we need to classify in our case it is Damage
        # self.add_class("dataset", 1, "Damage")
        self.add_class(DATASET, 1, "Femea") #2 e 3 instares
        self.add_class(DATASET, 2, "Macho")
        self.add_class(DATASET, 3, "Linfa") #1 instar femea

        LENGHT = len(IMAGES_LIST)
        
        OFFSET = math.floor(LENGHT * 0.8)
        # print(OFFSET)
        for i in range(LENGHT):
            
            
            # regra dos 80/20 corrigido em 13.06.2023
            if not is_train: #it is test (last 20%)
                if i < OFFSET:
                    # print(f'passs>> {i}')
                    continue
            else: # it is training
                if i > OFFSET:
                    # print(f'passs>> {i}')
                    continue

            # using add_image function we pass image_id, image_path and ann_path so that the current
            # image is added to the dataset for training or testing
            
            self.add_image(source=DATASET,image_id=IMAGE_ID_LIST[i], path=IMAGES_LIST[i], annotation=ANNOTATIONS_LIST[i])
            

    # function used to extract bouding boxes from annotated files
    def extract_boxes(self, filename):

        # you can see how the images are annotated we extracrt the width, height and bndbox values
        # <annotation>
        # <size>
        #       <width>640</width>
        #       <height>360</height>
        #       <depth>3</depth>
        # </size>
        # <object>
        #          <name>damage</name>
        #          <pose>Unspecified</pose>
        #          <truncated>0</truncated>
        #          <difficult>0</difficult>
        #          <bndbox>
        #                 <xmin>315</xmin>
        #                 <ymin>160</ymin>
        #                 <xmax>381</xmax>
        #                 <ymax>199</ymax>
        #          </bndbox>
        # </object>
        # </annotation>
        
        # used to parse the .xml files
        tree = ElementTree.parse(filename)
        
        # to get the root of the xml file
        root = tree.getroot()
        
        # we will append all x, y coordinated in boxes
        # for all instances of an onject
        boxes = list()
        
        # we find all attributes with name bndbox
        # bndbox will exist for each ground truth in image
        for box in root.findall('.//object'):
            name = box.find('.//name').text
            xmin = int(box.find('.//xmin').text)
            ymin = int(box.find('.//ymin').text)
            xmax = int(box.find('.//xmax').text)
            ymax = int(box.find('.//ymax').text)
            coors = [xmin, ymin, xmax, ymax, name]
            boxes.append(coors)
        
            

        # extract width and height of the image
        width = int(root.find('.//size/width').text)
        height = int(root.find('.//size/height').text)
        
        # return boxes-> list, width-> int and height-> int 
        return boxes, width, height
    
    # this function calls on the extract_boxes method and is used to load a mask for each instance in an image
    # returns a boolean mask with following dimensions width * height * instances
    def load_mask(self, image_id):
        
        # info points to the current image_id
        info = self.image_info[image_id]
        
        # we get the annotation path of image_id which is dataset_dir/annots/image_id.xml
        path = info['annotation']
        
        # we call the extract_boxes method(above) to get bndbox from .xml file
        boxes, w, h = self.extract_boxes(path)
        
        # we create len(boxes) number of masks of height 'h' and width 'w'
        masks = zeros([h, w, len(boxes)], dtype='uint8')
        
        
        class_ids = list()
        
        # we loop over all boxes and generate masks (bndbox mask) and class id for each instance
        # masks will have rectange shape as we have used bndboxes for annotations
        # for example:  if 2.jpg have three objects we will have following masks and class_ids
        # 000000000 000000000 000001110 
        # 000011100 011100000 000001110
        # 000011100 011100000 000001110
        # 000000000 011100000 000000000
        #    1         1          1    <- class_ids
        for i in range(len(boxes)):
            box = boxes[i]
            row_s, row_e = box[1], box[3]
            col_s, col_e = box[0], box[2]
            
            
            if (box[4] == 'Femea'):
                masks[row_s:row_e, col_s:col_e, i] = 1
                class_ids.append(self.class_names.index('Femea'))
            elif(box[4] == 'Macho'):
                masks[row_s:row_e, col_s:col_e, i] = 2
                class_ids.append(self.class_names.index('Macho')) 
            elif(box[4] == 'Linfa'):
                masks[row_s:row_e, col_s:col_e, i] = 3
                class_ids.append(self.class_names.index('Linfa'))
           
        
        # return masks and class_ids as array
        return masks, asarray(class_ids, dtype='int32')
    
    # this functions takes the image_id and returns the path of the image
    def image_reference(self, image_id):
        info = self.image_info[image_id]
        return info['path']



# %%
# cochonilha configuration class, you can change values of hyper parameters here
class CchConfig(Config):
    # name of the configuration
    NAME = "cochonilha"
    
    # ch class + background class
    NUM_CLASSES = 3 + 1
    
    # steps per epoch and minimum confidence
    STEPS_PER_EPOCH = 25
    
    # learning rate and momentum
    LEARNING_RATE=0.002
    LEARNING_MOMENTUM = 0.8
    
    # regularization penalty
    WEIGHT_DECAY = 0.0001
    
    # image size is controlled by this parameter
    IMAGE_MIN_DIM = 300
    
    # validation steps
    VALIDATION_STEPS = 50
    
    # number of Region of Interest generated per image
    Train_ROIs_Per_Image = 55
    
    # RPN Acnhor scales and ratios to find ROI
    RPN_ANCHOR_SCALES = (16, 32, 48, 64, 128)
    RPN_ANCHOR_RATIOS = [0.5, 1, 1.5]


# %%
# prepare train set
train_set = CchDataset()
#  organize data
(images, annotations, path_ids) = train_set.prepare_dataset(IMAGE_DIR,ANNOTATIONS_DIR)

train_set.load_dataset(images,annotations,path_ids,True)
train_set.prepare()

# validation/test
test_set = CchDataset()
test_set.load_dataset(images,annotations,path_ids,False)
test_set.prepare()

# load damage config
config = CchConfig()

# define the model
model = MaskRCNN(mode='training', model_dir=MODEL, config=config)


# %%

# load weights mscoco model weights
weights_path = os.path.join(ROOT,'mask_rcnn_coco.h5')

# load the model weights
model.load_weights(weights_path, 
                   by_name=True, 
                   exclude=["mrcnn_class_logits", "mrcnn_bbox_fc","mrcnn_bbox", "mrcnn_mask"])



# %% [markdown]
# <h1>Treine o modelo</h1>

# %%

# start the training of model
# you can change epochs and layers (head or all)
model.train(train_set, 
            test_set, 
            learning_rate=config.LEARNING_RATE, 
            epochs=10, 
            layers='heads')


