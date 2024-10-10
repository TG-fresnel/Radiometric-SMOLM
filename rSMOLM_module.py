# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 13:45:15 2024

@author: Tobias
"""
import numpy as np 
#%%from ProcessingTools

def Vflip(array):
    """ Flip the image vertically.
    """
    return np.flip(array, 0)

def take_from(array, width, center):
    """ Return a square within the array.

    Arguments:
        width (int): width of the centered square
        center (tupple): coordinates of the center of the array to extract (y,x)
    """
    if array.ndim == 2:
        length_y, length_x = array.shape
    elif array.ndim == 3:
        length_y, length_x,_ = array.shape
    k = width // 2
    n = width % 2
    step_y, step_x = 0, 0
    up = max(0, center[0] - k + 1 - n + step_y)
    down = min(length_y, center[0] + k + 1 + step_y)
    left = max(0, center[1] - k + 1 - n + step_x)
    right = min(length_x, center[1] + k + 1 + step_x)

    return array[up: down, left: right]


#%%
def extract_channels_from_single_frame(img,N_channels,box_length,ROI_centers):
    
    """
    Extracts each channels data from image


    Parameters
    ----------
    img : 2d array
        2d input image.
    N_channels : int
        number of channels.
    box_length : int
        box length of each channels subimages.
    ROI_centers : 2d array
        array containing the centers of each channel, dimensions of array 2 x N_channels.


    Returns
    -------
    data : 3d-array
        3d array containing the data of all channels, dimension 0 corresponds to number of channels, dimensions 1 and 2 are spatial.

    """
    data = np.zeros([N_channels,box_length, box_length])

    for i in range(N_channels):
      
        data[i,:,:] =  take_from(img, box_length, ROI_centers[:,i])
                       
        
    return data 


def load_data_from_single_frame(img_in, ROI_props, extra_box_pixels = 0):
    """
    Loads each channels data from image based on ROI_props

    Parameters
    ----------
    img_in : 2d array
        2d input image
    ROI_props : dict
        dict that contains: 'centroid-0','centroid-1','axis_major_length'.
    extra_box_pixels : TYPE, optional
        extra margin pixels when extracting channels. The default is 0.

    Returns
    -------
    data : 3d-array
      3d array containing the data of all channels, dimension 0 corresponds to number of channels, dimensions 1 and 2 are spatial.

    """

    
    img = Vflip(1.*img_in )
    N_channels = len(ROI_props)
    
    #rounded integers of the center coordinates of each channel
    ROI_centers = np.round(np.array([ROI_props['centroid-0'],ROI_props['centroid-1']])).astype(int)

    #Make uniform box length choosing the largest
    box_length = ( np.round(np.max([ROI_props['axis_major_length']])).astype(int) + extra_box_pixels)

    data = extract_channels_from_single_frame(img,N_channels,box_length,ROI_centers)
        
    return data 


def extract_masks(ROI_masks, ROI_props, extra_box_pixels = 0):
    """
    

    Parameters
    ----------
    ROI_masks : 2d-integer array
        Mask array where each channels mask is indicated by an integer.
    ROI_props : dict
        dict that contains: 'centroid-0','centroid-1','axis_major_length'.
    extra_box_pixels : TYPE, optional
        extra margin pixels when extracting channels. The default is 0.

    Returns
    -------
    2d-integer array
        DESCRIPTION.

    """
    
    N_channels = len(ROI_props)
    
    #rounded integers of the center coordinates of each channel
    ROI_centers = np.round(np.array([ROI_props['centroid-0'],ROI_props['centroid-1']])).astype(int)

    #Make uniform box length choosing the largest
    box_length = ( np.round(np.max([ROI_props['axis_major_length']])).astype(int) + extra_box_pixels)

    
    extracted_masks = extract_channels_from_single_frame(ROI_masks,4,box_length,ROI_centers)
    
    for i in range(N_channels):
 
            extracted_masks[i,...] = extracted_masks[i,...]==(i+1)
            
    return extracted_masks.astype(int)


























