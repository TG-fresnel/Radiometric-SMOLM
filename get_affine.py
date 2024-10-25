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
import pickle

from os.path import join

# Personal modules

from rSMOLM_module import *

#%%
data_path = 'data/'
ROI_props_path = join(data_path,'ROI.pkl')
ROI_masks_path = join(data_path,'ROI_masks.npy') 

img_path =  join(data_path,'Beads.tif')

affine_dict_path = join(data_path,'affine_dict.pkl')

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

#%%create dict that has keys identifying each channel like 'channel0', 'channel1' etc and contains as items a list of the conv-maks for each channel
conv_masks_dict = create_conv_mask_dict(conv_masks_orentation_list,
                                        conv_masks_minor_major_axis_ratio_list,
                                        minimal_axis_length,
                                        maximal_axis_length)


#%%

#use the generated conv-masks to detcetd PSFs by convolution
PSF_data = detect_psf_areas_all_channels(img_data,
                                         conv_masks_dict,
                                         show_final_plot = show_final_plot_psf_detection,
                                         show_all_plots = show_all_plots_psf_detection)

#the convolution approach leads to aretefacts at the borders therefore the errroded masks are initialized.
erroded_masks = erode_masks_with_disk(extracted_masks,footprint_radius=erosion_footprint_radius)

#Appyling erroded masks to exclude border regions
PSF_data = PSF_data * erroded_masks

#perform opening to get rid of small regions
PSF_data = open_masks_area(PSF_data,min_size=min_size_PSF)

#%%


#store information about each detected PSF dict of dataframes for each channel, this includes performing a gauss fit 
dict_PSF_dfs = extract_psf_data_for_all_channels(PSF_data,img_data,12) ###add option to perform gauss fit or not

#apply filter to each df of dict_PSF_dfs
dict_PSF_dfs_filtered = filter_dict_PSF_dfs(dict_PSF_dfs, limits_dict)

#generate a dict of affine tranformations 
dict_affine_transforms = get_affine_all_channels(dict_PSF_dfs_filtered)
    
    
#%%show affine, change source channel to 'channel2' or 'channel3' to see for the other channels

plot_affine_transformation(dict_PSF_dfs, 
                           dict_affine_transforms,
                           target_channel_name = 'channel0',
                           source_channel_name='channel1',
                           img_data = img_data[0])
#%% save dict_affine

with open(affine_dict_path, 'wb') as pickle_file:
    pickle.dump(dict_affine_transforms, pickle_file)  
       


    


