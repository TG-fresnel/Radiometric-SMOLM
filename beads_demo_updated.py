# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 13:44:45 2024

@author: Tobias
"""

#%%

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

from os.path import join,split

# Personal modules

from rSMOLM_module import *

#%%
extra_box_pixels= 0
N_channels =4

conv_masks_orentation_list = [0,0,np.pi*2/3,-np.pi*2/3]
conv_masks_minor_major_axis_ratio_list = [1,0.66,0.66,0.66]
minimal_axis_length = 2
maximal_axis_length = 6
#%%
ROI_path = 'data/'
ROI_props_path = ROI_path+ 'ROI.pkl'
ROI_masks_path = ROI_path+'ROI_masks.npy'

img_path = 'data/Beads.tif'

##%

ROI_props = pd.read_pickle(ROI_props_path)
ROI_masks = np.load(ROI_masks_path)
img_in = plt.imread(img_path)
#%%

#load all channels in one 3d-array
data_unmasked =  load_data_from_single_frame(img_in, ROI_props, extra_box_pixels = extra_box_pixels)

#extract masks from ROI_masks into same shape as data
extracted_masks = extract_masks(ROI_masks, ROI_props, extra_box_pixels = extra_box_pixels)

#multiply data with masks to apply masking 
data_masked = data_unmasked*extracted_masks
















