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

# affine_dict_path = join(data_path,'affine_dict.pkl')

#%%
extra_box_pixels= 0
N_channels = 6

conv_masks_orentation_list = [0,2*np.pi*1/5,2*np.pi*2/5,2*np.pi*3/5,2*np.pi*4/5,2*np.pi*5/5] - np.deg2rad(30)
conv_masks_minor_major_axis_ratio_list = [1,0.66,0.66,0.66,0.66,0.66]
minimal_axis_length = 2
maximal_axis_length = 6
min_size_PSF = 5
erosion_footprint_radius = 20
PSF_windowsize= 16

#criteria for filtering PSFs
limits_dict = {
    
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
#option to turn on plots of psf detection
show_final_plot_psf_detection = True #generates one plot per channel that indicated the detected psfs
show_all_plots_psf_detection = True #genreates a figure of subplots per channel that shows the used conv masks and their effects


#%%load data 

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

#%%
conv_masks_orentation_list = ROI_props['angles'].values
conv_masks_dict = create_conv_mask_dict(conv_masks_orentation_list,
                                        conv_masks_minor_major_axis_ratio_list,
                                        minimal_axis_length,
                                        maximal_axis_length)


PSF_data = detect_psf_areas_all_channels(img_data,
                                         conv_masks_dict,
                                         show_final_plot = show_final_plot_psf_detection,
                                         show_all_plots = show_all_plots_psf_detection)

#%%
erroded_masks = erode_masks_with_disk(extracted_masks,footprint_radius=erosion_footprint_radius)

#Appyling erroded masks to exclude border regions
PSF_data = PSF_data * erroded_masks

#perform opening to get rid of small regions
PSF_data = open_masks_area(PSF_data,min_size=min_size_PSF)
#%%

dict_PSF_dfs = extract_psf_data_for_all_channels(PSF_data,img_data,12)


#%%
show_PSFs_channel(img_data,dict_PSF_dfs,0)


#%%
show_PSFs_channel(img_data,dict_PSF_dfs,0,'centroid-1','centroid-0')
















