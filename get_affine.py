#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 13:48:46 2024

@author: tobias
"""


#%%
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

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
min_size_PSF = 5
erosion_footprint_radius = 20
PSF_windowsize= 16
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
conv_masks_dict = create_conv_mask_dict(conv_masks_orentation_list,
                                        conv_masks_minor_major_axis_ratio_list,
                                        minimal_axis_length,
                                        maximal_axis_length)



#%%
PSF_data = detect_psf_areas_all_channels(data_masked,conv_masks_dict,show_final_plot = False,show_all_plots=False)
erroded_masks = erode_masks_with_disk(extracted_masks,footprint_radius=erosion_footprint_radius)

PSF_data = PSF_data * erroded_masks

PSF_data = open_masks_area(PSF_data,min_size=min_size_PSF)

#%%
limits_all_channels = {
    
    'channel0' : {
        'amplitude': (1000, None),
        'intensity_mean': (1000, None)},
    
    'channel1'  : {
        'amplitude': (800, None),
        'intensity_mean': (800, None)},
    
    'channel2'  : {
        'amplitude': (1000, None),
        'intensity_mean': (800, None)},
    
    'channel3'  : {
        'amplitude': (800, None),
        'intensity_mean': (800, None)}
    }


dict_PSF_dfs = extract_psf_data_for_all_channels(PSF_data,data_masked,12,limits_all_channels)


dict_affine_transforms = get_affine_all_channels(dict_PSF_dfs)


#%%

with open('affine_transforms.pkl', 'wb') as pickle_file:
    pickle.dump(dict_affine_transforms, pickle_file)        
        
#%%
    
with open('affine_transforms.pkl', 'rb') as pickle_file:
    dict_affine_transforms_loaded = pickle.load(pickle_file)

    

    
#%%
plot_affine_transformation(dict_PSF_dfs,'channel0','channel2',img_data = data_masked[0])


#%%
show_PSFs_channel(data_masked,dict_PSF_dfs,0)
#%%checking all overlap of all detcted points
for column,df in dict_PSF_dfs.items():
    coords = get_coords(df)
    plt.scatter(coords[:,1],coords[:,0],marker = 'x')
plt.imshow(data_unmasked[0])

#%%




