import cv2
from object_classification import detect_image

from models import *
from utils import *

import os, sys, time, datetime, random
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

config_path='config/yolov3.cfg'
weights_path='config/yolov3.weights'
class_path='config/coco.names'
img_size=416
conf_thres=0.8
nms_thres=0.4


# Load model and weights
model = Darknet(config_path, img_size=img_size)
model.load_weights(weights_path)
model.eval()
classes = utils.load_classes(class_path)
Tensor = torch.FloatTensor

vid = cv2.VideoCapture(0)

while(True):
        ret,frame = vid.read()
        
        #TODO        
        #frame = cv2.resize(frame,(224,224))
        img = Image.fromarray((frame * 255).astype(np.uint8))
        imgX = img.size[0] #Width
        imgY = img.size[1] #Height
        
        detections = detect_image(img)
        
        # Get bounding-box colors
        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i) for i in np.linspace(0, 1, 20)]

        img = np.array(img)

        fig, ax = plt.subplots(1, figsize=(12,9))
        ax.imshow(frame)

        pad_x = max(img.shape[0] - img.shape[1], 0) * (img_size / max(img.shape))
        pad_y = max(img.shape[1] - img.shape[0], 0) * (img_size / max(img.shape))
        unpad_h = img_size - pad_y
        unpad_w = img_size - pad_x

        if detections is not None:
            unique_labels = detections[:, -1].cpu().unique()
            n_cls_preds = len(unique_labels)
            bbox_colors = random.sample(colors, n_cls_preds)
            # browse detections and draw bounding boxes
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                box_h = ((y2 - y1) / unpad_h) * img.shape[0]
                box_w = ((x2 - x1) / unpad_w) * img.shape[1]
                y1 = ((y1 - pad_y // 2) / unpad_h) * img.shape[0]
                x1 = ((x1 - pad_x // 2) / unpad_w) * img.shape[1]
                color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
                bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor='none')
                ax.add_patch(bbox)
                plt.text(x1, y1, s=classes[int(cls_pred)], color='white', verticalalignment='top',
                        bbox={'color': color, 'pad': 0})
                
                #Calcule le centre de la cible
                xCenter = box_w.item()/2+x1.item()
                yCenter = box_h.item()/2+y1.item()
                plt.scatter(xCenter, yCenter, color='red')

        plt.axis('off')

        plt.show()
        cv2.imshow('frame', frame)
        
        if cv2.waitKey(500) & 0xFF == ord('q'):
            break
        
vid.release()
cv2.destroyAllWindows()