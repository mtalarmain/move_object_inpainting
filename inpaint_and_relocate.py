#!/usr/bin/env python
# import required libraries
import cv2
import numpy as np
import argparse
from time import time
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image
import torch
from PIL import Image
import requests
from transformers import SamModel, SamProcessor
import matplotlib.pyplot as plt
import random


### Manually draw boudingbow with user mouse and exctract its coordinates ###

# now let's initialize the list of reference point
ref_point = []

def shape_selection(event, x, y, flags, param):
    # grab references to the global variables
    global ref_point, crop

    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being performed
    if event == cv2.EVENT_LBUTTONDOWN:
        ref_point = [(x, y)]

    # check to see if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates and indicate that
        # the cropping operation is finished
        ref_point.append((x, y))

        # draw a rectangle around the region of interest
        cv2.rectangle(image, ref_point[0], ref_point[1], (0, 255, 0), 2)
        cv2.imshow("image", image)


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", default="landscape_img.jpg", help="Path to the image")
ap.add_argument("-p", "--prompt", default=None, help="Prompt that describe the image to inpaint")
args = vars(ap.parse_args())

# load the image, clone it, and setup the mouse callback function
path_img = args["image"]
image = cv2.imread(path_img)
clone = image.copy()
cv2.namedWindow("image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("image", 1000, 700) 
cv2.setMouseCallback("image", shape_selection)


# keep looping until the 'q' key is pressed
while True:
    # display the image and wait for a keypress
    cv2.imshow("image", image)
    key = cv2.waitKey(1) & 0xFF

    # press 'r' to reset the window
    if key == ord("r"):
        image = clone.copy()

    # if the 'c' key is pressed, break from the loop
    elif key == ord("c"):
        print('Top left point:', ref_point[0], 'Bottom right point:', ref_point[1])
        # Top left and botton right point of the box. (x1, y1, x2, y2): x1: the x coordinate of the top left point of the input box. y1: the y coordinate of the top left point of the input box. x2: the x coordinate of the bottom right point of the input box. y2: the y coordinate of the bottom right point of the input box
        input_boxes = [[[ref_point[0][0], ref_point[0][1], ref_point[1][0], ref_point[1][1]]]]
        break

# close all open windows
cv2.destroyAllWindows()

### Segmentation of object inside bounding box with SAM ###

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

model = SamModel.from_pretrained("facebook/sam-vit-base")
processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

raw_image = Image.open(path_img).convert("RGB")
img_size = raw_image.size

inputs = processor(raw_image, input_boxes=input_boxes, return_tensors="pt")#.to("cuda")
outputs = model(**inputs, multimask_output=False)
masks = processor.image_processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu())[0]

masks_reshaped = np.moveaxis(masks[0].numpy(), 0, -1).astype('float64')
masked = np.ma.masked_where(masks_reshaped == 0, masks_reshaped)

# visualize the predicted masks
path_mask = path_img[:-4] + '_mask' + path_img[-4:]
cv2.imwrite(path_mask, masks_reshaped*255)

### Perform Inpainting based on SAM segmentation mask ###

pipe = AutoPipelineForInpainting.from_pretrained("diffusers/stable-diffusion-xl-1.0-inpainting-0.1", torch_dtype=torch.float16, variant="fp16").to("cuda")

image = load_image(path_img).resize((1024, 1024))
mask_image = load_image(path_mask).resize((1024, 1024))

if args["prompt"] is None:
    prompt = str(input("Enter a detailed prompt that describe the image:")) 
else:
    prompt = args["prompt"]
generator = torch.Generator(device="cuda").manual_seed(0)

image_inpainting = pipe(
  prompt=prompt,
  image=image,
  mask_image=mask_image,
  guidance_scale=8.0,
  num_inference_steps=20,  # steps between 15 and 30 work well for us
  strength=0.99,  # make sure to use `strength` below 1.0
  generator=generator,
).images[0].resize(img_size)

path_inpainting = path_img[:-4] + '_inpainting' + path_img[-4:]
image_inpainting.save(path_inpainting)

### Relocate object to the desired place on image ###

img = Image.open(path_img)
img_inpaint = Image.open(path_inpainting).resize(img_size)
mask = Image.open(path_mask)
back_im = img_inpaint.copy()
back_im.paste(img, (240, 40), mask)
path_repositioning = path_img[:-4] + '_move_object' + path_img[-4:]
back_im.save(path_repositioning)


### Compute centroid mask ###

img = cv2.imread(path_mask)
# convert image to grayscale image
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
 
# convert the grayscale image to binary image
ret,thresh = cv2.threshold(gray_image,127,255,0)
 
# calculate moments of binary image
M = cv2.moments(thresh)
 
# calculate x,y coordinate of center
cX = int(M["m10"] / M["m00"])
cY = int(M["m01"] / M["m00"])

# put text and highlight the center
cv2.circle(img, (cX, cY), 5, (255, 0, 255), -1)
cv2.putText(img, "centroid", (cX - 25, cY - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
 
path_centroid = path_mask[:-4] + '_centroid' + path_mask[-4:]
cv2.imwrite(path_centroid, img)

### Compute new rectangle bounding box coordinate ###

src_gray = cv2.blur(gray_image, (3,3))

threshold = 100
    
canny_output = cv2.Canny(src_gray, threshold, threshold * 2)
    
    
contours, _ = cv2.findContours(canny_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
      
contours_poly = [None]*len(contours)
boundRect = [None]*len(contours)
for i, c in enumerate(contours):
    contours_poly[i] = cv2.approxPolyDP(c, 3, True)  
    boundRect[i] = cv2.boundingRect(contours_poly[i]) 

list_area = []   
for i in range(len(contours)):
    area = boundRect[i][2] * boundRect[i][3]
    list_area.append(area)
# keep rectangle with larger area
idx = np.argmax(list_area)
color = (random.randint(0,256), random.randint(0,256), random.randint(0,256))
cv2.rectangle(img, (int(boundRect[idx][0]), int(boundRect[idx][1])), \
          (int(boundRect[idx][0]+boundRect[idx][2]), int(boundRect[idx][1]+boundRect[idx][3])), color, 2)


input_boxes = [[[int(boundRect[idx][0]), int(boundRect[i][1]), int(boundRect[idx][0]+boundRect[idx][2]), int(boundRect[idx][1]+boundRect[idx][3])]]]
print('new bounding box coordinate:',input_boxes) 
path_rectangle = path_centroid[:-4] + '_rectangle' + path_centroid[-4:]
cv2.imwrite(path_rectangle, img)