# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 13:44:45 2024

@author: Tobias
"""

#%%

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

from os.path import join,split

# Personal modules

from rSMOLM_module import *

#%%
extra_box_pixels= 10
N_channels =4

conv_masks_orentation_list = [0,0,np.pi*2/3,-np.pi*2/3]
conv_masks_minor_major_axis_ratio_list = [0.66,1,0.66,0.66]
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

#%%
def get_conv_masks(orientation,minor_major_axis_ratio,mininmal_axis_length,maximal_axis_length):
    """
    

    Parameters
    ----------
    orientation : list
        list containing the orientation of each channels PSF with respect to the vertical.
        Length of list must correspond with number of channels.
    minor_major_axis_ratio : list
        list containing the ratio between the minor axis of the ellipse and the major axis.
        Length of list must correspond with number of channels.
    mininmal_axis_length : int
        Major axis length of the smallest ellipse.
    maximal_axis_length : int
       Major axis length of the biggest ellipse.

    Returns
    -------
    conv_masks : list
        List of masks

    """
    

    conv_masks = []
    
    #l lenght of major axis, index to assign values to conv_masks array
    for i,l in enumerate(range(mininmal_axis_length,maximal_axis_length)):
        
        major_axis_length   = l
        minor_axis_length = np.round(l * minor_major_axis_ratio).astype(int)
        print(major_axis_length,minor_axis_length)
        conv_masks.append( ellips_perimeter_mask(major_axis_length,minor_axis_length,  orientation))
        
    return conv_masks
    

conv_masks_dict = {}


for i in range(N_channels):
    

    print(i)
    conv_masks_orentation = conv_masks_orentation_list[i] 
    conv_masks_minor_major_axis_ratio = conv_masks_minor_major_axis_ratio_list[i]
    
    
    conv_masks_dict[f'channel{i}'] = get_conv_masks(conv_masks_orentation,conv_masks_minor_major_axis_ratio,2,6)


























