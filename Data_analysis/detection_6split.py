# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 13:44:45 2024

@author: Tobias
"""

#%%

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import pickle

from os.path import join,split

# Personal modules

from rSMOLM_module import *

#%%
extra_box_pixels= 0
N_channels = 6
 

erosion_footprint_radius = 20
 
#%%
data_path = 'C:/Users/Tobias/Documents/Data/HexBFP/20250303/Registration/'
ROI_props_path = join(data_path,'ROI.pkl')
ROI_masks_path = join(data_path,'ROI_masks.npy') 
img_path =  join(data_path,'Beads00001.tif')


##%
ROI_props = pd.read_pickle(ROI_props_path)
ROI_masks = np.load(ROI_masks_path)
img_in = plt.imread(img_path)

#%%prepare data

#load all channels in one 3d-array
data_unmasked =  load_data_from_single_frame(img_in, ROI_props, extra_box_pixels = extra_box_pixels)

#extract masks from ROI_masks into same shape as data
extracted_masks = extract_masks(ROI_masks, ROI_props, extra_box_pixels = extra_box_pixels)

#multiply data with masks to apply masking 
img_data = data_unmasked*extracted_masks

#%%

fig,_ = show_multiple_channels(img_data)