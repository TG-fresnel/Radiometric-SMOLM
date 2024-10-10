# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 12:13:56 2024

@author: Tobias
"""




#%%
from os.path import join,split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
from matplotlib.patches import ConnectionPatch,Rectangle
from matplotlib.collections import PatchCollection
from skimage.morphology import erosion,disk, area_opening
from skimage.measure import regionprops,regionprops_table,label,centroid
from functools import partial 
from scipy.optimize import curve_fit
from scipy.signal import convolve2d,correlate2d
from scipy.spatial import distance_matrix
from scipy.spatial.distance import pdist, squareform
from scipy.ndimage import distance_transform_edt
from skimage.transform import warp, AffineTransform
from skimage.measure import ransac
from skimage.filters import threshold_otsu
from skimage.segmentation import watershed
from sympy.ntheory import factorint

# Personal modules

from functions_collection import *


#%%

extra_box_pixels= 10
ROI_path = 'data/'


props = pd.read_pickle(ROI_path+"ROI.pkl")
ROI_masks = np.load(ROI_path+'ROI_masks.npy')

#Read the centroids 
centroids_roi = np.round(np.array([props['centroid-0'],props['centroid-1']])).astype(int)

#Make uniform box length choosing the largest
box_length = ( int(np.round(np.max([props['axis_major_length']]))) 
              + extra_box_pixels)

corners = centroids_roi - box_length//22
#%%
img_path = 'data/Beads.tif'

img_in = plt.imread(img_path)


img = Vflip(1.*img_in )

data = np.zeros([4,box_length, box_length])

edge_buffer = 0
erosion_mask = disk(edge_buffer)

for i in range(4):
    if edge_buffer > 0:
        ROI_mask = erosion(ROI_masks==(i+1),erosion_mask)
    else:
        ROI_mask = ROI_masks==(i+1)    
    masked_img = img * ROI_mask
    data[i,:,:] =  (take_from(masked_img,
                                 box_length,[centroids_roi[0][i],centroids_roi[1][i]])
                    *disk(box_length//2, [box_length, box_length]))


#%%
num_channels = 4
errosion_size = 9 
min_size = 3 


plot_all = False
plot_final = True

maj_min_ax_ratio = [0.66,1,0.66,0.66]
orentation = [0,0,np.pi*2/3,-np.pi*2/3]

labeled = np.zeros_like(data).astype(dtype=int)
errosion_masks = get_erroded_masks(ROI_masks,props,footprint_size =errosion_size,extra_box_pixels= 10)
    
    
for i in range(4):

    conv_masks = [ellips_perimeter_mask(k,int(k*maj_min_ax_ratio[i]),orentation[i]) for k in range(2,6)]

    labeled[i,...] = find_psfs_and_clean( data[i,...],
                                         conv_masks,
                                         ROI_masks,
                                         props,      
                                         i,
                                         errosion_masks[i,...],
                                         min_size,
                                         plot_all=plot_all,
                                         plot_final=plot_final)
  




#%% store psfs information of all channels in a single multi index df

window_size = 9

#select from the 100 brightest psfs all that have no neighbours within 10 pixels
all_channels_df = create_psfs_df(labeled,data,window_size)


psfs_ordered_df = all_channels_df.sort_values(by=['source','avg_intensity'],ascending=False)
bright_psfs_df = psfs_ordered_df.groupby('source').apply(lambda x: x.nlargest(100, 'avg_intensity'))
bright_seperated_psfs_df = bright_psfs_df.loc[((bright_psfs_df['min_dist'] > 10))]


bright_seperated_psfs_df_gaus = gaus_fit_from_df(bright_seperated_psfs_df)
# bright_seperated_psfs_df_gaus = bright_seperated_psfs_df_gaus[np.isnan(bright_seperated_psfs_df['gauss_x0'].values)==False]

link_all_channels(bright_seperated_psfs_df_gaus,position_method = 'centroid')



show_psfs_from_df(data,bright_seperated_psfs_df_gaus,show_matches=True,show_psfs=True,show_gauss = True,psf_window_size=window_size )


# %%
