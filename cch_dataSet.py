"""
ESTA CLASSE SERVE PARA CARREGAR O DATA-SET, CARREGA A IMAGEM,
SEU ID E RESPECTIVA ANOTAÇÃO XML, TAMBÉM CARREGA MASCARA E BOUNDING BOX

*CLASSE BASE DERIVADA DE MASK_RCNN OU MRCNN
"""


import math
import os
from xml.etree import ElementTree
from numpy import asarray, zeros
from mrcnn.utils import Dataset
from mrcnn.config import Config
# ROOT
ROOT = os.getcwd()
DATASET = os.path.join(ROOT,'dataset')



class CchDataset(Dataset):
    #  prepare dataset and ignore images without annotations
    def prepare_dataset(self,IMAGE_DIR, ANNOTATIONS_DIR):

        images = []
        annotations = []
        image_id_list = []
        images_dir_list = os.listdir(IMAGE_DIR)
        # TODO fase de teste dados serão sequenciais
        # shuffle(images_dir_list)
        
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
                    # print(f'test passs>> {i}')
                    continue
            else: # it is training
                if i > OFFSET:
                    # print(f' training passs>> {i}')
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
