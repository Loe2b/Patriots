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

def detect_image(img):
    # scale and pad image
    ratio = min(img_size/img.size[0], img_size/img.size[1])
    imw = round(img.size[0] * ratio)
    imh = round(img.size[1] * ratio)
    img_transforms = transforms.Compose([ transforms.Resize((imh, imw)),
         transforms.Pad((max(int((imh-imw)/2),0), max(int((imw-imh)/2),0), max(int((imh-imw)/2),0), max(int((imw-imh)/2),0)),
                        (128,128,128)),
         transforms.ToTensor(),
         ])
    # convert image to Tensor
    image_tensor = img_transforms(img).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input_img = Variable(image_tensor.type(Tensor))
    # run inference on the model and get detections
    with torch.no_grad():
        detections = model(input_img)
        detections = utils.non_max_suppression(detections, 80, conf_thres, nms_thres)
    return detections[0]
# load image and get detections
img_path = "images/animalMalCadre.jpg"
prev_time = time.time()
img = Image.open(img_path)

imgX = img.size[0] #Width
imgY = img.size[1] #Height

detections = detect_image(img)
inference_time = datetime.timedelta(seconds=time.time() - prev_time)
print ('Inference Time: %s' % (inference_time))

# Get bounding-box colors
cmap = plt.get_cmap('tab20b')
colors = [cmap(i) for i in np.linspace(0, 1, 20)]

img = np.array(img)

fig, ax = plt.subplots(1, figsize=(12,9))
ax.imshow(img)

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

#Save image
plt.savefig(img_path.replace(".jpg", "-det.jpg"), bbox_inches='tight', pad_inches=0.0)

#Calcule le milieu du champ de vision == le centre de l'image
newImg = Image.open(img_path.replace(".jpg", "-det.jpg"))
imgXCenter = newImg.size[0]/2
imgYCenter = newImg.size[1]/2
print("Centre de l'image en : X = ", imgXCenter, " Y = ", imgYCenter)
fig, ax = plt.subplots(1, figsize=(12,9))
ax.imshow(newImg)
plt.axis('off')
plt.scatter(imgXCenter, imgYCenter, color='blue')

#Save image
plt.savefig(img_path.replace(".jpg", "-det.jpg"), bbox_inches='tight', pad_inches=0.0)

#Calcule la distance à parcourir pour recentrer la cible
imgRatio = newImg.size[0]/imgX #Afin de convertir la position du centre de la cible 
                                #en accord avec les nouvelles proportions de l'image

#L'image change de taille pendant le traitement, on calcule le changement de taille
xCenter *= imgRatio
yCenter *= imgRatio

dst = ((imgXCenter - xCenter)**2+(imgYCenter - yCenter)**2)**0.5

print("La cible est en : X = ", int(xCenter), " Y = ", int(yCenter))

#Distance négative en X = tourner vers la gauche ; positive = tourner vers la droite
#Distance négative en Y = tourner vers le bas    ; positive = tourner vers le haut
toTurnX = xCenter-imgXCenter
toTurnY = imgYCenter - yCenter
print("Distance a parcourir : en X = ", toTurnX, " en Y = ", toTurnY)
print("Distance totale a parcourir : ", dst, "\n\n\n")

plt.show()

#Calcul de l'angle que la caméra doit parcourir
angleCameraX = 54
pixelByDegreesX = newImg.size[0] / angleCameraX
toTurnAngleX = round(toTurnX / pixelByDegreesX)

angleCameraY = 41
pixelByDegreesY = newImg.size[1] / angleCameraY
toTurnAngleY = round(toTurnY / pixelByDegreesY)

print("La caméra doit tourner de ", toTurnAngleX, "° horizontalement et de ", toTurnAngleY, "° verticalement")

#Caracteristiques de la camera:
#2592x1944 pixels
#54x41 degrés (angles de vue)



