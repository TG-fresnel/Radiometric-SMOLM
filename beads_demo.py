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
N_channels =4

conv_masks_orentation_list = [0,0,np.pi*2/3,-np.pi*2/3]
conv_masks_minor_major_axis_ratio_list = [1,0.66,0.66,0.66]
minimal_axis_length = 2
maximal_axis_length = 6
PSF_windowsize = 12
erosion_footprint_radius = 20
min_size_PSF = 5
mind_dist_union = 3

#dictionary to specify limits on which
limits_all_channels = {
    'channel0' : {
        'x0': (0, PSF_windowsize),
        'y0': (0, PSF_windowsize)
        },

    
    'channel1'  : {
        'x0': (0, PSF_windowsize),
        'y0': (0, PSF_windowsize)
        },

    
    'channel2'  : {
        'x0': (0, PSF_windowsize),
        'y0': (0, PSF_windowsize)
        },

    
    'channel3'  : {
        'x0': (0, PSF_windowsize),
        'y0': (0, PSF_windowsize)
        },

    }

show_final_plot_psf_detection = False
show_all_plots_psf_detection = False 
#%%
data_path = 'data/'
ROI_props_path = join(data_path,'ROI.pkl')
ROI_masks_path = join(data_path,'ROI_masks.npy') 
img_path =  join(data_path,'Beads.tif')
affine_dict_path = join(data_path,'affine_dict.pkl')

##%
ROI_props = pd.read_pickle(ROI_props_path)
ROI_masks = np.load(ROI_masks_path)
img_in = plt.imread(img_path)

with open(affine_dict_path, 'rb') as pickle_file:
    dict_affine_transforms = pickle.load(pickle_file)

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


#%%PSF detection

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

#%%Gauss-fit and filtering 

#store information about each detected PSF dict of dataframes for each channel
#this includes performing a gauss fit 
dict_PSF_dfs = extract_psf_data_for_all_channels(PSF_data,img_data,PSF_windowsize) ###add option to perform gauss fit or not

#apply filter to each df of dict_PSF_dfs
dict_PSF_dfs_filtered = filter_dict_PSF_dfs(dict_PSF_dfs, limits_all_channels)

#%%perform union of all channels PSFs coordinates

union_coords_list = dict_PSF_dfs_union(dict_PSF_dfs_filtered,
                       dict_affine_transforms,max_dist=mind_dist_union)


#%%display results of union

show_multiple_channels(img_data,
                       coords=union_coords_list, 
                       dict_affine_transforms = dict_affine_transforms,
                       main_channel = 0,
                       show_colorbar = True,
                       )

#%%perform box integration 
#at the coordintes of union_coords_list, 
#the coordintates are tranformed accordint to the affine tranformations

box_integration_intensities_df = box_integration_all_channels_to_df(img_data = img_data,
                                       coords = union_coords_list,
                                       dict_affine_transforms = dict_affine_transforms,
                                       PSF_windowsize = 12)

#%%show ratios as histogram 
intensity_ratios_hist(box_integration_intensities_df)
