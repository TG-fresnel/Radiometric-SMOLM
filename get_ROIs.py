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
from skimage.morphology import convex_hull_object
from skimage.measure import find_contours,label,regionprops_table


# Personal module

from rSMOLM_module import *

#%%
extra_box_pixels= 10
img_path = 'data/Img_for_ROI.tif'

analysis_path =  'data/'

#%%

imgROI = Vflip(1.*plt.imread(img_path))

threshold = 0.9
# Now I binarize to define the areas of interest
ROI = binarize(imgROI,threshold=threshold, radius = 2)
ROI = convex_hull_object(ROI)


#%%
# Getting their properties
ROI_masks = label(ROI)
properties = ('centroid','orientation','axis_major_length','axis_minor_length','bbox')
ROI_props_dict = regionprops_table(ROI_masks,properties=properties)
ROI_props = pd.DataFrame.from_dict(ROI_props_dict)
#%%

# Saving their properties into a file
dum = pd.DataFrame(props)
dum.to_pickle(analysis_path+'ROI.pkl')

#saving masks 
np.save(analysis_path+'ROI_masks.npy',ROI_masks)

