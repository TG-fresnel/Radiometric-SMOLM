# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 13:46:02 2025

@author: Storm4
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt




fontsize = 15
plt.rcParams["font.size"] = str(fontsize)


from rSMOLM_module import *
from utility_functions import *

img_path = 'C:/Users/Tobias/Documents/Data/HexBFP/20250303/Pupil/'
analysis_path = 'C:/Users/Tobias/Documents/Data/HexBFP/20250303/Pupil/'
# img_path = "D:/BFP splitting/20250108/Single color beads/Green/Pupil/"
# analysis_path = "D:/BFP splitting/20250108/Single color beads/Green/"
#Newest camera ORCA inverts the image fully (x,y)->(-x,-y), to counter the negative magnification
flip_flag = False #Normally loaded images are flipped to counter the automatic flipping;
#however, for the pupil study this is not needed, we want to be in the true pûpil orientation


pupil_full = load_data(img_path+"FullPupil00000.tif")
pupil_full = counts_to_photons(pupil_full)
pupil_closed = load_data(img_path+"ClosedPupil00000.tif")
pupil_closed= counts_to_photons(pupil_closed)

if flip_flag == False:
    #Undo the default flip
    pupil_full = np.flip( np.flip(pupil_full, 0), 1)
    pupil_closed = np.flip( np.flip(pupil_closed, 0), 1)


threshold = 0.8
# Binarize to define the areas of interest
mask_full = binarize(pupil_full, threshold=threshold, radius=3, smin=1000)
mask_full = convex_hull_object(mask_full)
mask_full = label(mask_full)
properties = (
    "centroid",
    "orientation",
    "axis_major_length",
    "axis_minor_length",
    "bbox",
)
full_props_dict = regionprops_table(mask_full, properties=properties)
full_props = pd.DataFrame.from_dict(full_props_dict)

mask_closed = binarize(pupil_closed, threshold=threshold, radius=3, smin=1000)
mask_closed = convex_hull_object(mask_closed)
mask_closed = label(mask_closed)
properties = (
    "centroid",
    "orientation",
    "axis_major_length",
    "axis_minor_length",
    "bbox",
)
closed_props_dict = regionprops_table(mask_closed, properties=properties)
closed_props = pd.DataFrame.from_dict(closed_props_dict)

full_center = (np.round(full_props["centroid-0"][0]),np.round(full_props["centroid-1"][0]))
closed_center = (np.round(closed_props["centroid-0"][0]),np.round(closed_props["centroid-1"][0]))

print("Centroid of full pupil = ", full_center )
print("Centroid of closed pupil = ", closed_center )

L_full = (full_props["axis_major_length"]+full_props["axis_minor_length"])/2
L_closed = (closed_props["axis_major_length"]+closed_props["axis_minor_length"])/2

effective_pupil = L_closed[0]/L_full[0]
print("Effective NA ratio = ", effective_pupil)

plt.figure()
plt.imshow(mask_full)
plt.scatter(full_center[1] , full_center[0],label=str(full_center), marker='+', color='green')
plt.xlabel(r"$x$ [pix]")
plt.ylabel(r"$y$ [pix]")
plt.title("Full Pupil")
plt.legend()
plt.tight_layout()
plt.savefig(analysis_path+"FullPupil.svg", transparent=True)
plt.show()

plt.figure()
plt.imshow(mask_closed)
plt.scatter(closed_center[1] , closed_center[0],label=str(closed_center), marker='+', color='green')
plt.xlabel(r"$x$ [pix]")
plt.ylabel(r"$y$ [pix]")
plt.title("Closed Pupil, ratio="+str(np.round(effective_pupil,2)))
plt.legend()
plt.tight_layout()
plt.savefig(analysis_path+"ClosedPupil.svg", transparent=True)
plt.show()


pupil_props = pd.concat([full_props, closed_props])

save_ROI = input("Do you want to save this result(y/n):") == "y"
if save_ROI:
    pupil_props.to_pickle(analysis_path + "pupil.pkl")
    # saving masks
    print("Parameters saved !")
    
# #%%
# import cv2
# image = pupil_full.astype(np.uint8)
# image = cv2.resize(image, (1152, 1152))    
# # image = cv2.GaussianBlur(image,(25,25),0)

# ret, thresh = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY_INV)

# cv2.imshow('Binary image', thresh)
# cv2.destroyAllWindows()
# contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
                               
# img = cv2.cvtColor(image ,cv2.COLOR_GRAY2BGR)
# # draw contours on the original image
# contour_img = cv2.drawContours(image=img, contours=contours, contourIdx=-1, color=(0, 255,0),
#                  thickness=20, lineType=cv2.LINE_AA)


# cv2.imshow("Result", 1*img)


# #%%
# img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)  #add this line

# cv2.drawContours(img, contours, -1, (0, 255,0), 2)
# cv2.imshow("Result", img)