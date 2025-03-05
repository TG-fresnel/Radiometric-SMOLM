#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 18:35:54 2024

@author: tobias


This file creates the ROI_masks.npy and ROI.pkl files
"""
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt


# Personal module

from rSMOLM_module import *

#%%

extra_box_pixels= 0 #additional margin of masks
img_path = 'C:/Users/Tobias/Documents/Data/HexBFP/20250303/Registration/FOV00000.tif' #file of image to define ROI
analysis_path = 'C:/Users/Tobias/Documents/Data/HexBFP/20250303/Registration/' #directory where ROI.pkl and ROI_masks.npy will be saved

#%%load and binarize image

imgROI = Vflip(1.*plt.imread(img_path))

threshold = 0.9
# Now I binarize to define the areas of interest
ROI = binarize(imgROI,threshold=threshold, radius = 2)
ROI = convex_hull_object(ROI)


#%% get ROI properties 
ROI_masks = label(ROI)
properties = ('centroid','orientation','axis_major_length','axis_minor_length','bbox')
ROI_props_dict = regionprops_table(ROI_masks,properties=properties)
ROI_props = pd.DataFrame.from_dict(ROI_props_dict)

#order ROIs in sucha a way that the radial channel is 0 and the azimuthal ones are ordered clockwise
ROI_props, ROI_masks = auto_order_ROI(ROI_props,ROI_masks)



#%%save ROI_props dataframe as a pkl file and ROI_masks as a npy file
# Saving their properties into a file


ROI_props.to_pickle(analysis_path+'ROI.pkl')
#saving masks 
np.save(analysis_path+'ROI_masks.npy',ROI_masks)


plt.figure()
plt.imshow(ROI_masks)

plt.colorbar()
plt.tight_layout()
plt.savefig(analysis_path+"Labeled_Masks.svg", transparent=True)


