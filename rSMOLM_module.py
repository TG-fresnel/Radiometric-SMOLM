# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 13:45:15 2024

@author: Tobias
"""
import numpy as np 
import matplotlib.pyplot as plt

from scipy.signal import convolve2d
from skimage.draw import ellipse_perimeter
from skimage.morphology import binary_erosion,binary_dilation,remove_small_holes,remove_small_objects,square,convex_hull_object,disk,erosion
from skimage.measure import find_contours,label,regionprops_table
from scipy.spatial import distance_matrix
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


#%% functinos related to ROIs
def auto_order_ROI(ROI_props, ROI_masks):
    """
    Automatically reorder regions of interest (ROIs) by setting the central ROI as the 0th index 
    and ordering the remaining ROIs in a clockwise manner, starting from a vertical line.

    Parameters
    ----------
    ROI_props : pandas.DataFrame
        A DataFrame containing the properties of the ROIs.
    ROI_masks : numpy.ndarray
        A 2-d array containing the masks of the ROIs.

    Returns
    -------
    ROI_props : pandas.DataFrame
        The reordered DataFrame of ROI properties with the central ROI first and the remaining ROIs
        ordered clockwise.
    ROI_masks : numpy.ndarray
        The relabeled and reordered ROI masks following the new order.


    """

    # Compute the center coordinates of the mask
    center_coords = np.array(ROI_masks.shape).reshape((1, 2)) / 2

    # Extract the centroid coordinates of each ROI
    centroids = get_coords(ROI_props)

    # Get the current order of ROI indices
    channels_index = list(ROI_props.index)

    # Calculate the Euclidean distance of each ROI centroid to the center
    distance_center = distance_matrix(center_coords, centroids)

    # Shift centroid coordinates relative to the center
    shifted_centroids = centroids - center_coords

    # Convert Cartesian coordinates of ROIs to polar angles (radians)
    angles = xy_to_polar_angle(shifted_centroids[:, 1], shifted_centroids[:, 0])

    # Shift angles to reference them from the vertical line (y-axis)
    angles = add_angles(angles, np.pi / 2)

    # Identify the central ROI (the one closest to the center)
    radial_channel_idx = np.argmin(distance_center)

    # Sort the remaining ROIs based on their angular positions (clockwise order)
    angular_sorted_idx = np.argsort(angles)

    # Remove the central ROI from the angular sorted list
    azimuthal_channel_ordered_idx = remove_by_value(angular_sorted_idx, radial_channel_idx)

    # Concatenate the central ROI index with the ordered indices
    new_idx = np.concatenate(([radial_channel_idx], azimuthal_channel_ordered_idx))

    # Reorder the ROI properties and masks based on the new index
    ROI_props, ROI_masks = order_ROI(ROI_props, ROI_masks, new_idx)

    return ROI_props, ROI_masks


def order_ROI(ROI_props, ROI_masks, new_idx):
    """
    Reorder the properties and masks of regions of interest (ROIs) based on a new index.

    Parameters
    ----------
    ROI_props : pandas.DataFrame
        A DataFrame containing the properties of the ROIs.
    ROI_masks : numpy.ndarray
        A 2-d array containing the masks of the ROIs.
    new_idx : array-like
        An array of indices specifying the new order for the ROIs.

    Returns
    -------
    ROI_props : pandas.DataFrame
        The reordered DataFrame of ROI properties with the new index applied.
    ROI_masks : numpy.ndarray
        The relabeled ROI masks based on the new index.

    """

     
    ROI_props = ROI_props.iloc[new_idx]#apply new index
    ROI_props = ROI_props.reset_index(drop=True)

    new_idx_for_mask = np.concatenate(([0],new_idx+1))

    ROI_masks = relabel_masks(ROI_masks, new_idx_for_mask)
    
    
    return ROI_props,ROI_masks

    
def relabel_masks(arr, new_order):
    
    
    """
    Replaces values in a 2D array based on a new order of values.

    Parameters:
    -----------
    arr : np.ndarray
        A 2D numpy array where the values are among [0, 1, 2, 3].
    new_order : list
        A list of length 4 representing the new order of values. 
        The index corresponds to the original value, and the value at each index 
        indicates what the original value should be replaced with.
        For example, if new_order = [0, 3, 1, 2]:
            - 0 remains 0
            - 1 becomes 3
            - 2 becomes 1
            - 3 becomes 2

    Returns:
    --------
    np.ndarray
        A new 2D numpy array with the values replaced according to the new order.

    Example:
    --------
    >>> arr = np.array([
            [0, 1, 2, 3],
            [3, 2, 1, 0],
            [1, 2, 3, 0]
        ])
    >>> new_order = [0, 3, 1, 2]
    >>> replace_values(arr, new_order)
    array([[0, 3, 1, 2],
           [2, 1, 3, 0],
           [3, 1, 2, 0]])
    """
    # Create a mapping from the current value to the new value
    mapping = {i: new_order[i] for i in range(len(new_order))}

    # Use numpy's vectorized function to replace the values efficiently
    vectorized_replace = np.vectorize(lambda x: mapping[x])

    # Apply the function to the input array
    return vectorized_replace(arr)


#%% functions related to loading image and extracting channels
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
            
    return extracted_masks.astype(bool)

#%%functions related to PSF detection
def get_conv_masks(orientation,minor_major_axis_ratio,mininmal_axis_length,maximal_axis_length):
    """
    Generate convolution masks based on the specified orientation and axis ratios of ellipses.

    Parameters
    ----------
    orientation : list of float
        A list containing the orientation of each channel's point spread function (PSF) with respect to the vertical.
        The length of the list must correspond to the number of channels.
    minor_major_axis_ratio : list of float
        A list containing the ratio between the minor axis and the major axis of the ellipses.
        The length of the list must correspond to the number of channels.
    minimal_axis_length : int
        The length of the major axis of the smallest ellipse.
    maximal_axis_length : int
        The length of the major axis of the largest ellipse.

    Returns
    -------
    conv_masks : list
        A list of masks generated for each channel based on the specified parameters. Each mask represents 
        an ellipse perimeter defined by the corresponding orientation, major axis length, and minor axis length.

    """
    

    conv_masks = []
    
    #l lenght of major axis, index to assign values to conv_masks array
    for i,l in enumerate(range(mininmal_axis_length,maximal_axis_length)):
        
        major_axis_length   = l
        minor_axis_length = np.round(l * minor_major_axis_ratio).astype(int)
        conv_masks.append( ellips_perimeter_mask(major_axis_length,minor_axis_length,  orientation))
        
    return conv_masks
    

def create_conv_mask_dict(orientation_list, minor_major_axis_ratio_list, minimal_axis_length, maximal_axis_length):
    """
    Create a dictionary of convolution masks for each channel based on the provided orientation
    and minor-to-major axis ratio of ellipses.

    Parameters
    ----------
    orientation_list : list of float
        A list containing the orientations of each channel's ellipses with respect to the vertical.
        The length of the list must correspond to the number of channels.
    minor_major_axis_ratio_list : list of float
        A list containing the ratios of the minor axis to the major axis for each channel's ellipses.
        The length of the list must correspond to the number of channels.
    minimal_axis_length : int
        The length of the major axis of the smallest ellipse.
    maximal_axis_length : int
        The length of the major axis of the largest ellipse.

    Returns
    -------
    conv_masks_dict : dict
        A dictionary where keys are channel identifiers (e.g., 'channel0', 'channel1', etc.)
        and values are the convolution masks generated for each channel.
    
    Raises
    ------
    ValueError
        If the lengths of the orientation_list and minor_major_axis_ratio_list do not match.
    """
    
    # Validate that both lists have the same length
    if len(orientation_list) != len(minor_major_axis_ratio_list):
        raise ValueError("Both lists must have the same length.")

    N_channels = len(orientation_list)
    conv_masks_dict = {}

    for channel_index in range(N_channels):
        # Get orientation and axis ratio for the current channel
        channel_orientation = orientation_list[channel_index] 
        channel_minor_major_axis_ratio = minor_major_axis_ratio_list[channel_index]

        # Create and store the convolution mask in the dictionary
        conv_masks_dict[f'channel{channel_index}'] = get_conv_masks(
            channel_orientation,
            channel_minor_major_axis_ratio,
            mininmal_axis_length=minimal_axis_length,
            maximal_axis_length=maximal_axis_length
        )

    return conv_masks_dict


def detect_psf_areas(image, masks, show_all_plots=False, show_final_plot=False):
    """
    Identify areas in the input image that correspond to point spread functions (PSFs) 
    by convolving the image with a set of masks and detecting regions where the convolution results consistently decrease.

    Parameters
    ----------
    image : numpy.ndarray
        The input image to be convolved. It should be a 2D array representing grayscale image data.
    masks : list of numpy.ndarray
        A list of 2D arrays representing convolution masks (point spread functions) to be applied to the image.
    show_all_plots : bool, optional
        If True, plots the masks, the convolved images, and the areas where the convolved images decrease.
        Default is False.
    show_final_plot : bool, optional
        If True, plots the original image with contours indicating the detected PSF areas.
        Default is False.

    Returns
    -------
    numpy.ndarray
        A boolean array where True indicates areas of the image that are identified as PSFs,
        based on the condition that each convolved result is less than the previous one.

    Notes
    -----
    The function generates a stack of convolved images and computes the differences between successive
    convolutions. Areas where each convolved image is less than the previous one are considered to 
    represent PSFs.
    """
    
    # Initialize an array to hold the convolved image stack
    convolved_images = np.zeros((*image.shape, len(masks) + 1))
    convolved_images[..., 0] = image  # Assign the original image to the first layer

    # Perform convolution with each mask
    for i, mask in enumerate(masks):
        convolved_images[..., i + 1] = convolve2d(image, mask, mode='same')

    # Compute the differences between successive convolved images
    convolution_differences = np.diff(convolved_images, axis=2)
    
    # Identify areas where all differences are less than 0, indicating potential PSFs
    psf_mask = np.all(convolution_differences < 0, axis=2)

    # Plot all masks and convolved images if requested
    if show_all_plots: 
        fig1, axes1 = plt.subplots(3, len(masks), layout='constrained', figsize=(3 * len(masks), 9))
        
        for i, mask in enumerate(masks):
            axes1[0, i].imshow(mask)
            axes1[0, i].set_title(f'Mask {i + 1}')
            axes1[0, i].axis('off')

            axes1[1, i].imshow(convolved_images[..., i + 1])
            axes1[1, i].set_title(f'Convolved Image {i + 1}')
            axes1[1, i].axis('off')

            axes1[2, i].imshow(np.all(convolution_differences[..., :i + 1] < 0, axis=2))
            axes1[2, i].set_title(f'Detected PSF')
            axes1[2, i].axis('off')

    # Plot the final result indicating detected PSF areas if requested
    if show_final_plot:
        fig2, axes2 = plt.subplots(1, 1, layout='constrained', figsize=(6, 6))
        axes2.imshow(image, cmap='gray')
        axes2.contour(psf_mask, alpha=0.5, colors='yellow', linewidths=1)
        axes2.set_title('Detected PSF Areas')
        axes2.axis('off')

    return psf_mask



def detect_psf_areas_all_channels(data, masks_dict, show_all_plots=False, show_final_plot=False):
    """
    Detect point spread function (PSF) areas across multiple channels in a 3D image stack
    by applying corresponding masks to each channel and using the PSF detection algorithm.

    Parameters
    ----------
    data : numpy.ndarray
        A 3D array where each slice along the third dimension represents a separate channel of image data.
    masks_dict : dict
        A dictionary where keys are channel identifiers (e.g., 'channel0') and values are lists of masks
        to be applied to the corresponding channel for PSF detection.
    show_all_plots : bool, optional
        If True, plots the PSF detection results for all channels. Default is False.
    show_final_plot : bool, optional
        If True, plots the final detection results overlaid on the original images. Default is False.

    Returns
    -------
    numpy.ndarray
        A 3D array of the same shape as the input `data`, where each slice represents the PSF
        detection results for the corresponding channel.
    """
    
    N_channels = data.shape[0]
    detected_psfs = np.zeros_like(data,dtype=bool)
    
    
    for channel_idx in range(N_channels):
        # Extract the image for the current channel
        image = data[channel_idx,...]
        masks = masks_dict[f'channel{channel_idx}']
        
        # Detect PSF areas for the current channel
        detected_psfs[channel_idx,...] = detect_psf_areas(image, masks, show_all_plots=show_all_plots, show_final_plot=show_final_plot)
        
    return detected_psfs



def open_masks_area(masks, min_size):
    """
    Perform area opening on a 3D stack of masks to remove small objects.

    Parameters
    ----------
    masks : numpy.ndarray
        A 3D array where each slice corresponds to a different binary mask to be opened.
    min_size : int
        Minimum size of objects to retain in the masks. Objects smaller than this will be removed.

    Returns
    -------
    numpy.ndarray
        A 3D array of the same shape as the input `masks`, containing the opened masks.
    """
    
    opened_masks = np.zeros_like(masks, dtype=masks.dtype)  # Initialize the output array

    # Process each mask in the 3D stack
    for channel_idx in range(masks.shape[0]):
        # Apply area opening to the current mask and store it in the output array
        opened_masks[channel_idx, ...] = remove_small_objects(masks[channel_idx, ...],min_size=min_size)

    return opened_masks



def erode_masks_with_disk(masks, footprint_radius):
    """
    Erode a 3D stack of masks using a disk-shaped footprint of a given radius.

    Parameters
    ----------
    masks : numpy.ndarray
        A 3D array where each slice corresponds to a different mask to be eroded.
    footprint_radius : int
        Radius of the disk used for the erosion operation.

    Returns
    -------
    numpy.ndarray
        A 3D array of the same shape as the input `masks`, containing the eroded masks.
    """
    
    # Create the erosion mask using the specified footprint radius
    erosion_mask = disk(footprint_radius)
    eroded_masks = np.zeros_like(masks)  # Initialize the output array with the same shape as input masks

    # Process each mask in the 3D stack
    for channel_idx in range(masks.shape[0]):
        # Apply erosion to the current mask and store it in the output array
        eroded_masks[channel_idx, ...] = erosion(masks[channel_idx, ...], erosion_mask)

    return eroded_masks

        

def ellips_perimeter_mask(maj_ax,min_ax,orientation=0):
    
    
   
    
    img = np.zeros((maj_ax*2+1, maj_ax*2+1), dtype=np.uint8)
    rr, cc = ellipse_perimeter(maj_ax, maj_ax,maj_ax,min_ax,orientation,shape=img.shape)
    img[rr, cc] = 1

    return img[min(rr):max(rr)+1,min(cc):max(cc)+1]/img.sum()


#%% utilty functions         
def xy_to_polar_angle(x, y):
    """
    Convert Cartesian coordinates (x, y) to polar angles in radians.

    Args:
    x (array-like): X-coordinates.
    y (array-like): Y-coordinates.

    Returns:
    numpy.ndarray: Polar angles in radians, in the range [0, 2π].
    """
    x = np.array(x)
    y = np.array(y)
    
    # Calculate angles using np.arctan2, which accounts for quadrant and handles x=0
    angles = np.arctan2(y, x)
    
    # Adjust negative angles to be within the range [0, 2π]
    angles = np.where(angles < 0, angles + 2*np.pi, angles)
    
    return angles

def add_angles(angle1, angle2):
    """
    Adds two angles and wraps the result within the range of 0 to 360 degrees.
    
    If angle1 is a list, the function will add angle2 to each element in the list.
    The result is wrapped using modulus 360 to ensure it remains within the valid range of angles.

    Parameters:
    - angle1 (float or list of floats): The first angle or list of angles in degrees.
    - angle2 (float): The second angle to be added in degrees.

    Returns:
    - float or list of floats: The resulting angle(s) wrapped within the range [0, 360) degrees.
    """
    def wrap_angle(angle):
        """Helper function to wrap a single angle within [0, 360) degrees."""
        return np.mod(angle,np.pi*2)
    
    if isinstance(angle1, list):
        # Add angle2 to each element in the list and wrap them
        return [wrap_angle(a + angle2) for a in angle1]
    else:
        # Add two single angles and wrap the result
        return wrap_angle(angle1 + angle2)

def remove_by_value(arr_in,val):
    """
    Remove the first occurrence of a specified value from the input array.

    Parameters
    ----------
    arr_in : list or array-like
        The input array from which the value will be removed.
    val : any
        The value to be removed from the array.

    Returns
    -------
    numpy.ndarray
        A new array with the first occurrence of `val` removed.
    """
    
    arr_in = np.array(arr_in)
    
    index = np.argwhere(arr_in==val)
    arr_out = np.delete(arr_in, index[0][0])
    
    return arr_out
    
def get_coords(df,x_name = 'centroid-0',y_name = 'centroid-1'):
    """
    

    Parameters
    ----------
    df : pandas.core.frame.DataFrame
        Dataframe that contains x and y coordinates 
    x_name : string, optional
        Name of the column of the x coordinate. The default is 'centroid-0'.
    y_name : string, optional
        Name of the column of the y coordinate. The default is 'centroid-1'.

    Returns
    -------
    numpy.array 
        Array of dimension (n,2). With n beeing the number of rows in the original Dataframe

    """
    
    return np.array([df[x_name],df[y_name]]).T
    













