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

from matplotlib.patches import ConnectionPatch,Rectangle
from matplotlib.collections import PatchCollection

from skimage.morphology import erosion,disk, area_opening
from skimage.measure import regionprops,regionprops_table,label,centroid

from scipy.optimize import curve_fit
from scipy.signal import convolve2d,correlate2d
from scipy.spatial import distance_matrix
from scipy.spatial.distance import pdist, squareform
from scipy.ndimage import distance_transform_edt

from skimage.transform import warp, AffineTransform
from skimage.measure import ransac
from skimage.filters import threshold_otsu
from skimage.segmentation import watershed

# Personal modules

from rSMOLM_module import *

#%%
extra_box_pixels= 10
N_channels =4


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




























