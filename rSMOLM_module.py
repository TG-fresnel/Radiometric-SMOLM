# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 13:45:15 2024

@author: Tobias
"""
import numpy as np 

from scipy.signal import convolve2d
from skimage.draw import ellipse_perimeter
from skimage.morphology import binary_erosion,binary_dilation,remove_small_holes,remove_small_objects,square
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

def binarize(image, threshold=1, radius=1, smin=1000):
    """ The following function binarizes the image and then eliminates
	small holes and/or small objects.

	Arguments:
	    threshold (float): multiplicative factor with respect the mean of the array
	    radius (float): radius of the erosion and dilation steps
	    smin (int):  minimum elements for a hole or an object to not be removed
	"""

    # First we threshold the image with respect its mean and the given factor
    mask = (image > (threshold * np.mean(image)));
    # Erode to eliminatte noise
    mask = binary_erosion(mask, square(radius));
    # Dilate back
    mask = binary_dilation(mask, square(radius));
    # Remove small holes within the windows
    mask = remove_small_holes(mask, smin)
    # Remove small objects within the windows
    mask = remove_small_objects(mask, smin)
    return mask


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


def conv_psf_finder(img,masks):
    convolved_img_stack = np.zeros((*img.shape,len(masks)+1))
    convolved_img_stack[...,0] = img
    
    for i,mask in enumerate(masks):
           convolved_img_stack[...,i+1] =  convolve2d(img,mask,mode='same')
           
    stack_diff = np.diff(convolved_img_stack, axis=2)
    final = np.all(stack_diff< 0, axis=2)
    #compact way to check each convolved array is smaller thasn the previous one 

    return final     



def ellips_perimeter_mask(maj_ax,min_ax,orientation=0):
    
   
    
    img = np.zeros((maj_ax*2+1, maj_ax*2+1), dtype=np.uint8)
    rr, cc = ellipse_perimeter(maj_ax, maj_ax,maj_ax,min_ax,orientation,shape=img.shape)
    img[rr, cc] = 1

    return img[min(rr):max(rr)+1,min(cc):max(cc)+1]/img.sum()



















