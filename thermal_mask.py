# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 17:41:34 2021

@author: AdminFRI
"""
import os
import cv2
import shutil
import tifffile
import pandas as pd
import numpy as np
from PIL import Image

def file_detect(ext, directory=None):
    """Iterate through a directory and get all files of a specified extension"""
    files = os.listdir(directory)
    return([file for file in files if file.endswith(ext)])

# directory where unmasked TIF files are located
base = r"TIF/flight0000"

# directory in which to save masked TIF files

savedir = r"masked/flight0000"
if not os.path.exists(savedir):
    os.makedirs(savedir)

# n – counter variable
n = 0

# initialize array for mask
mask = np.zeros((512,640))
kernel = np.ones((3,3), np.float32)/9

# retrieve filenames of all TIF files in base directory
files = file_detect('.tif', base)


print("Calculating thermal mask...")
for filename in files:
    
    # Progress bar/percentage
    pct = round(n / len(files) * 100, 0)
    prog = int(pct / 5 - pct % 5 / 5)
    print(("#"*prog).ljust(20), "|", "{x}%".format(x=int(pct)).ljust(5), end='\r')
    
    # open image as an array
    fp = os.path.join(base, filename)
    img = np.array(tifffile.imread(fp))
    #img = (img - img.min()) / (img.max() - img.min())
    img = (img - img.mean()) / np.std(img)
    dst = cv2.filter2D(img, -1, kernel)
    
    
    # apply weighted average to mask and image
    mask = (mask * n + dst) / (n + 1)
    
    n += 1
    
print("\nDone calculating thermal mask.")

# Save mask to CSV format
pd.DataFrame(mask).to_csv(os.path.join(base, 'mask.csv'),header=False,index=False)

# apply mask
n = 0
print("Applying thermal mask...")
for filename in files:

    # Progress bar/percentage
    pct = round(n / len(files) * 100, 0)
    prog = int(pct / 5 - pct % 5 / 5)
    print(("#"*prog).ljust(20), "|", "{x}%".format(x=int(pct)).ljust(5), end='\r')
    
    # get original image
    img = Image.open(os.path.join(base, filename))
    values = np.array(img)
    
    # subtract mask from original
    masked_img = values - (mask * np.std(values))

    # make masked_img into an Image object with PIL
    masked_img = Image.fromarray(masked_img.astype("uint16"))
    
    # save de-masked image to output directory
    img.paste(masked_img)
    img.save(os.path.join(savedir, filename), "tiff")
    n += 1

print("\nDone applying thermal mask.")