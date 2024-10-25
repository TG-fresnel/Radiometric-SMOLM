# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 13:45:15 2024

@author: Tobias
"""
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

from scipy.signal import convolve2d
from skimage.draw import ellipse_perimeter
from skimage.morphology import binary_erosion,binary_dilation,remove_small_holes,remove_small_objects,square,convex_hull_object,disk,erosion
from skimage.measure import find_contours,label,regionprops_table,ransac
from skimage.transform import AffineTransform
from scipy.spatial import distance_matrix
from scipy.optimize import curve_fit
from tqdm import tqdm 

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


def extract_centered_square(array, width, center, return_anchor=False):
    """
    Extract a square subarray centered on specific coordinates from a 2D or 3D array.

    Parameters
    ----------
    array : ndarray
        The input array from which the subarray is extracted. Can be 2D or 3D.
    width : int
        The width of the square to extract. The square will be centered on the `center` coordinates.
    center : tuple of float
        The (y, x) coordinates of the center of the square. Can be floating point values,
        which will be rounded to the nearest integers.
    return_anchor : bool, optional
        If True, the function will also return the top-left corner coordinates (anchor point)
        of the extracted subarray as a tuple `(up, left)`. Default is False.

    Returns
    -------
    subarray : ndarray
        The extracted square subarray.
    (subarray, (up, left)) : tuple, optional
        If `return_anchor` is True, a tuple containing the subarray and the coordinates of the
        top-left corner (anchor) is returned.

    Examples
    --------
    >>> import numpy as np
    >>> arr = np.arange(25).reshape(5, 5)
    >>> extract_centered_square(arr, 3, (2, 2))
    array([[ 6,  7,  8],
           [11, 12, 13],
           [16, 17, 18]])

    >>> extract_centered_square(arr, 3, (2.7, 2.4), return_anchor=True)
    (array([[ 6,  7,  8],
            [11, 12, 13],
            [16, 17, 18]]), (1, 1))

    """

    if array.ndim == 2:
        length_y, length_x = array.shape
    elif array.ndim == 3:
        length_y, length_x, _ = array.shape

    # Round the center coordinates to nearest integers
    center_y = round(center[0])
    center_x = round(center[1])
    
    k = width // 2
    n = width % 2
    
    up = max(0, center_y - k)
    down = min(length_y, center_y + k + n)
    left = max(0, center_x - k)
    right = min(length_x, center_x + k + n)

    if return_anchor:
       
        return array[up:down, left:right],(up,left)

    else:
  
        return array[up:down, left:right]


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
    and ordering the remaining ROIs in a clockwise manner, with respect to the negative y-axis (up).

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

    # Convert Cartesian coordinates of ROIs to polar angles (radians),
    angles = xy_to_polar_angle(shifted_centroids[:, 0], -shifted_centroids[:, 1])
    # Shift angles to reference them from the vertical line (y-axis)
    angles = add_angles(angles, np.pi)



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

    for channel_idx in range(N_channels):

        data[channel_idx,:,:] =  extract_centered_square(img, box_length, ROI_centers[:,channel_idx])
                       
        
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
    ROI_centers = np.array([ROI_props['centroid-0'],ROI_props['centroid-1']])

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
        extra margin pixels. The default is 0.

    Returns
    -------
    2d-integer array
        DESCRIPTION.

    """
    
    N_channels = len(ROI_props)
    
    #rounded integers of the center coordinates of each channel
    ROI_centers = np.array([ROI_props['centroid-0'],ROI_props['centroid-1']])

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



def get_PSF_df(binary_psf_img, intensity_image, properties=('intensity_mean', 'centroid', 'num_pixels')):
    """
    Generate a DataFrame containing properties of regions in the input image.

    Parameters
    ----------
    binary_psf_img : numpy.ndarray
        The binary image that indicates which regions are considered PSFs
    intensity_image : numpy.ndarray
        The intensity image used to compute intensity-based properties.
    properties : tuple of str, optional
        The properties to measure for each region. Default is ('intensity_mean', 'centroid', 'num_pixels').

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the measured properties for each region in the image.
    """
    
    labeled_img = label(binary_psf_img)
    regionprops = regionprops_table(labeled_img,
                                    intensity_image=intensity_image,
                                    properties=properties)
    
    df = pd.DataFrame(regionprops)

    return df


def extract_and_fit_psf_data(binary_psf_img, intensity_image, PSF_windowsize):
    """
    Extract PSF centroids from a binary PSF image, fit Gaussian functions to these PSFs, 
    and combine the results into a single DataFrame.

    Parameters
    ----------
    binary_psf_img : numpy.ndarray
        A binary image where each PSF is represented by a distinct region.

    intensity_image : numpy.ndarray
        The corresponding intensity image from which the PSF characteristics will be derived.

    PSF_windowsize : int
        The size of the window around each PSF centroid used for fitting the Gaussian model.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the combined data, including PSF centroids and the fitted Gaussian parameters.
    """
    df_centroids = get_PSF_df(binary_psf_img, intensity_image)
    centroid_coords = get_coords(df_centroids)
    
    df_gauss = gauss_fit_all_PSFs(intensity_image, centroid_coords, PSF_windowsize)
    
    df = pd.concat([df_centroids, df_gauss], axis=1)
    
    return df


#%%Gaus fitting

def extract_PSF(img, windowsize, coords):
    """
    Extract the Point Spread Function (PSF) from an image at specified coordinates.

    Parameters
    ----------
    img : numpy.ndarray
        The input 2D image from which the PSF is to be extracted.
    windowsize : int
        The size of the window around the coordinates to extract the PSF.
    coords : tuple of two floats or ints
        The (x, y) coordinates around which the PSF will be extracted.

    Returns
    -------
    numpy.ndarray
        A cropped window from the image representing the PSF.
    """

    PSF = extract_centered_square(img, windowsize, coords)
    
    return PSF


def gaussian_2d(xy, amp, x0, y0, sigma_major, sigma_minor, theta, offset):
    """
    Compute a 2D Gaussian function with elliptical symmetry.

    Parameters
    ----------
    xy : tuple of numpy.ndarray
        A tuple containing two arrays (x, y), where `x` and `y` are the grid coordinates 
        where the Gaussian is evaluated.
    amp : float
        Amplitude of the Gaussian peak.
    x0 : float
        X-coordinate of the Gaussian center.
    y0 : float
        Y-coordinate of the Gaussian center.
    sigma_major : float
        Standard deviation of the Gaussian along the major axis.
    sigma_minor : float
        Standard deviation of the Gaussian along the minor axis.
    theta : float
        Rotation angle of the Gaussian's major axis from the x-axis (in radians).
    offset : float
        Constant offset to add to the Gaussian function.

    Returns
    -------
    numpy.ndarray
        Value of the 2D Gaussian function evaluated at each point (x, y).
    """
    x, y = xy
    
    x0 = float(x0)
    y0 = float(y0)
    a = (np.cos(theta)**2) / (2 * sigma_major**2) + (np.sin(theta)**2) / (2 * sigma_minor**2)
    b = -(np.sin(2 * theta)) / (4 * sigma_major**2) + (np.sin(2 * theta)) / (4 * sigma_minor**2)
    c = (np.sin(theta)**2) / (2 * sigma_major**2) + (np.cos(theta)**2) / (2 * sigma_minor**2)
    return offset + amp * np.exp(-(a * ((x - x0)**2) + 2 * b * (x - x0) * (y - y0) + c * ((y - y0)**2)))


def get_gauss_fit_initial_guess(img,
                                amp_guess = None,
                                x0_guess = None,
                                y0_guess = None,
                                sigma_major_guess = None,
                                sigma_minor_guess = None,
                                theta_guess = 0,
                                offset_guess = None):
    """
   Generate an initial guess for fitting a 2D Gaussian to an image.

   Parameters
   ----------
   img : numpy.ndarray
       2D image array used for generating the initial guesses.
   amp_guess : float, optional
       Initial guess for the amplitude of the Gaussian. If not provided, the maximum pixel value of `img` is used.
   x0_guess : float, optional
       Initial guess for the x-coordinate of the Gaussian center. If not provided, the center of the image is used.
   y0_guess : float, optional
       Initial guess for the y-coordinate of the Gaussian center. If not provided, the center of the image is used.
   sigma_major_guess : float, optional
       Initial guess for the standard deviation along the major axis. If not provided, 1/4th of the image size is used.
   sigma_minor_guess : float, optional
       Initial guess for the standard deviation along the minor axis. If not provided, 1/4th of the image size is used.
   theta_guess : float, optional
       Initial guess for the rotation angle of the Gaussian (in radians). Default is 0.
   offset_guess : float, optional
       Initial guess for the constant background offset. If not provided, the minimum pixel value of `img` is used.

   Returns
   -------
   numpy.ndarray
       A 1D array of length 7, containing the initial guesses for the Gaussian fit parameters:
       [amplitude, x0, y0, sigma_major, sigma_minor, theta, offset].
   """
    
    initial_guess = np.zeros(7)
    window_size = img.shape[0]
    
    
    if amp_guess:
        initial_guess[0] = amp_guess
    else:
        initial_guess[0] = img.max()
    
    
    if x0_guess:
        initial_guess[1] = x0_guess
    else:
        initial_guess[1] = window_size/2
        
    
    if y0_guess:
        initial_guess[2] = y0_guess
    else:
        initial_guess[2] = window_size/2
        
        
    if sigma_major_guess:
        initial_guess[3] = sigma_major_guess
    else:
        initial_guess[3] = window_size/4
        
        
    if sigma_minor_guess:
        initial_guess[4] = sigma_minor_guess
    else:
        initial_guess[4] = window_size/4
    
    
    if theta_guess:
        initial_guess[5] = theta_guess
    else:
        initial_guess[5] = 0
        
    if offset_guess:
        initial_guess[6] = offset_guess
    else:
        initial_guess[6] = img.min()
    
       
       
    return initial_guess
        
    

def fit_2d_gaussian(img,
                    amp_guess = None,
                    x0_guess = None,
                    y0_guess = None,
                    sigma_major_guess = None,
                    sigma_minor_guess = None,
                    theta_guess = 0,
                    offset_guess = None):
    """
    Fit a 2D Gaussian function with elliptical symmetry to an image.
    
    This function estimates the parameters of a 2D Gaussian function that best fits
    the input image using nonlinear least squares optimization. The Gaussian is defined
    by its amplitude, center coordinates, standard deviations along the major and minor
    axes, rotation angle, and offset.
    
    Parameters
    ----------
    img : numpy.ndarray
        2D array representing the image data to which the Gaussian function will be fitted.
    amp_guess : float, optional
        Initial guess for the amplitude of the Gaussian peak. If None, the maximum value in the image will be used.
    x0_guess : float, optional
        Initial guess for the x-coordinate of the Gaussian center. If None, the center of the image is used.
    y0_guess : float, optional
        Initial guess for the y-coordinate of the Gaussian center. If None, the center of the image is used.
    sigma_major_guess : float, optional
        Initial guess for the standard deviation along the major axis. If None, a quarter of the image size is used.
    sigma_minor_guess : float, optional
        Initial guess for the standard deviation along the minor axis. If None, a quarter of the image size is used.
    theta_guess : float, optional
        Initial guess for the rotation angle of the Gaussian's major axis (in radians). Default is 0.
    offset_guess : float, optional
        Initial guess for the constant offset added to the Gaussian function. If None, the minimum value in the image is used.
    
    Returns
    -------
    popt : numpy.ndarray
        Optimized parameters of the fitted Gaussian function in the order: 
        [amplitude, x0, y0, sigma_major, sigma_minor, theta, offset].
    pcov : 2D array
        The covariance of popt, representing the estimated uncertainty of the fitted parameters.
    """
        
    
    
    initial_guess = get_gauss_fit_initial_guess(img,
                                                amp_guess,
                                                x0_guess,
                                                y0_guess,
                                                sigma_major_guess,
                                                sigma_minor_guess,
                                                theta_guess,
                                                offset_guess)
    
    
    size_x, size_y = img.shape
    x = np.linspace(0, size_x - 1, size_x)
    y = np.linspace(0, size_y - 1, size_y)
    xx, yy = np.meshgrid(x, y)

    # Flatten the data and coordinate grids for curve fitting
    grid_coords = np.vstack((xx.ravel(), yy.ravel()))
    target_data = img.ravel()

    popt,pcov = curve_fit(gaussian_2d,grid_coords,target_data,p0=initial_guess)

                          
    return popt,pcov 

def gauss_fit_all_PSFs(img,coords,windowsize):
    
    """
    Fit a 2D Gaussian to multiple PSFs extracted from an image based on specified coordinates.
    
    Parameters
    ----------
    img : numpy.ndarray
        The input image from which PSFs are extracted.
    
    coords : numpy.ndarray
        A 2D array of shape (n, 2) containing the coordinates of the detected PSFs,
        where each row corresponds to [y_coordinate, x_coordinate].
    
    windowsize : int
        The size of the window to extract around each PSF coordinate.
    
    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the fitted parameters and their covariance for each PSF.
    """
    
    
    num_coords = len(coords)
    
    columns = ['amplitude', 
                'x0', 
                'y0', 
                'sigma_major', 
                'sigma_minor', 
                'theta', 
                'offset',
                'amplitude_cov', 
                'x0_cov', 
                'y0_cov', 
                'sigma_major_cov', 
                'sigma_minor_cov', 
                'theta_cov', 
                'offset_cov',
                'y0_global',
                'x0_global']
    
    
    gauss_fit_data = np.zeros((num_coords,len(columns)))
    
    for i, coord in enumerate(tqdm(coords)):
        
        
        PSF,(up,left) = extract_centered_square(img, windowsize, coord,return_anchor=True)
              
       
        
        try:
            popt,pcov = fit_2d_gaussian(PSF)
            
        except:
            gauss_fit_data[i,:] = np.nan
        else:
            gauss_fit_data[i,:7] = popt
            gauss_fit_data[i,7:14] =  np.sqrt(np.diag(pcov))
            gauss_fit_data[i,14] =  up + popt[2]
            gauss_fit_data[i,15] = left + popt[1]
        

    
    df = pd.DataFrame(columns=columns,data = gauss_fit_data)
    
    return df
        
        

def select_PSFs_from_df_old(df, 
              min_amp=None, max_amp=None, 
              min_x0=None, max_x0=None, 
              min_y0=None, max_y0=None, 
              min_sigma_major=None, max_sigma_major=None, 
              min_sigma_minor=None, max_sigma_minor=None, 
              min_theta=None, max_theta=None, 
              min_offset=None, max_offset=None):
    """
    ---old version-----
    will be replaced
    """
    
    # Dictionary to hold column limits
    limits = {
        'amplitude': (min_amp, max_amp),
        'x0': (min_x0, max_x0),
        'y0': (min_y0, max_y0),
        'sigma_major': (min_sigma_major, max_sigma_major),
        'sigma_minor': (min_sigma_minor, max_sigma_minor),
        'theta': (min_theta, max_theta),
        'offset': (min_offset, max_offset)
    }
    
    # Start with the full DataFrame
    filtered_df = df
    
    # Apply filters based on the provided limits
    for column, (min_val, max_val) in limits.items():
        if min_val is not None:
            filtered_df = filtered_df[filtered_df[column] >= min_val]
        if max_val is not None:
            filtered_df = filtered_df[filtered_df[column] <= max_val]
    
    return filtered_df



def select_PSFs_from_df(df, limits_dict):
    """
    ---old version-----
    will be replaced
    """
    
    # Dictionary to hold column limits

    
    # Start with the full DataFrame
    filtered_df = df
    
    # Apply filters based on the provided limits
    for column, (min_val, max_val) in limits_dict.items():
        if min_val is not None:
            filtered_df = filtered_df[filtered_df[column] >= min_val]
        if max_val is not None:
            filtered_df = filtered_df[filtered_df[column] <= max_val]
    
    return filtered_df

    



def PSF_gauss_plot(img):
    """
    Fit a 2D Gaussian to the input image and plot the original image alongside the fitted Gaussian.

    This function uses the `fit_2d_gaussian` function to estimate the parameters of a
    2D Gaussian function that best fits the provided image data. It then generates a
    plot displaying the original image and the corresponding fitted Gaussian image side by side.

    Parameters
    ----------
    img : numpy.ndarray
        2D array representing the image data to which the Gaussian function will be fitted.
    
    Returns
    -------
    None
        This function does not return any values. It generates plots directly.
    """


    popt,_ = fit_2d_gaussian(img)
    size_x, size_y = img.shape
    x = np.linspace(0, size_x - 1, size_x)
    y = np.linspace(0, size_y - 1, size_y)
    xx, yy = np.meshgrid(x, y)
    
    x0 = popt[1]
    y0 = popt[2]

    grid_coords = np.vstack((xx.ravel(), yy.ravel()))
    
    fit_data = gaussian_2d(grid_coords,*popt)
    fit_data = fit_data.reshape((size_x,size_y))
    
    plt.subplot(121)
    plt.imshow(img)
    plt.scatter(x0,y0,marker='x',color='red')
    
    plt.subplot(122)
    plt.imshow(fit_data)
#%% affine tranformation 


def get_affine(points_to_move,reference_points,min_samples=3,residual_threshold=2,max_trials=1000):
    """
    Estimate the affine transformation matrix that best aligns two sets of points using RANSAC.

    Parameters
    ----------
    points_to_move : array_like, shape (n, m)
        The coordinates of the points to be moved, where `n` is the number of points
        and `m` is the dimensionality (e.g., 2 for 2D points).

    reference_points : array_like, shape (k, m)
        The coordinates of the reference points, where `k` is the number of points
        and `m` is the dimensionality. Should be at least as many as `points_to_move`.

    min_samples : int, optional
        Minimum number of samples to use in each RANSAC iteration. Default is 3.

    residual_threshold : float, optional
        Maximum distance for a data point to be considered a fitting inlier. Default is 2.

    max_trials : int, optional
        Maximum number of RANSAC iterations to perform. Default is 1000.

    Returns
    -------
    model_robust : AffineTransform
        The estimated affine transformation model that maps `points_to_move` to `reference_points`.
        This is an instance of the `AffineTransform` class, containing the transformation parameters.

    Notes
    -----
    If the number of `reference_points` is less than the number of `points_to_move`, 
    the function selects the closest points from `points_to_move` to match the 
    `reference_points` using the distance matrix. Otherwise, it selects the closest 
    `reference_points` to the `points_to_move`.

    Raises
    ------
    ValueError
        If the input arrays do not have compatible shapes or if there are insufficient points
        for a robust estimation.
    """

    dist_M = distance_matrix(points_to_move, reference_points)
    
    if len(reference_points) < len(points_to_move):
        closest_idx = dist_M.argmin(axis=0)
        points_to_move = np.array(points_to_move)[closest_idx]
        reference_points = np.array(reference_points)
    
        
    else:
        closest_idx = dist_M.argmin(axis=1)
        reference_points = np.array(reference_points)[closest_idx]
        points_to_move = np.array(points_to_move)
        
    
    
    model_robust, inliers = ransac(
        (points_to_move,reference_points),
        AffineTransform,
        min_samples=min_samples,
        residual_threshold=residual_threshold,
        max_trials=max_trials
        )
    
    
    
    return model_robust


def get_affine_all_channels(dict_PSF_dfs,main_channel = 0,min_samples=3,residual_threshold=2,max_trials=1000):
    """
    Estimate affine transformations for all channels relative to a main channel.

    This function computes affine transformation matrices for a set of channels
    stored in a dictionary, aligning each channel to a specified main channel based
    on their spatial coordinates.

    Parameters
    ----------
    dict_PSF_dfs : dict of DataFrame
        A dictionary where keys are channel names (e.g., 'channel0', 'channel1', etc.)
        and values are DataFrames containing the coordinates for each channel.

    main_channel : int, optional
        The index of the main channel to which all other channels will be aligned. 
        Default is 0, which corresponds to 'channel0'.

    min_samples : int, optional
        Minimum number of samples to use in each RANSAC iteration. Default is 3.

    residual_threshold : float, optional
        Maximum distance for a data point to be considered a fitting inlier. Default is 2.

    max_trials : int, optional
        Maximum number of RANSAC iterations to perform. Default is 1000.

    Returns
    -------
    dict_affine_transforms : dict
        A dictionary containing the estimated affine transformations for each channel,
        with keys formatted as 'channelX->channelY' indicating the source and target channels.

    Raises
    ------
    KeyError
        If the specified main channel does not exist in the dictionary of DataFrames.
    ValueError
        If there are insufficient coordinates for the affine transformation estimation.
    """
    
    main_channel_name = f'channel{main_channel}'
    main_channel_df = dict_PSF_dfs[main_channel_name]
    
    channel_names = list(dict_PSF_dfs.keys())
    
    channels_to_link = channel_names
    channels_to_link.remove(main_channel_name)
    
    dict_affine_transforms = {}
    
    for channel in channels_to_link:
        
        channel_to_link_df = dict_PSF_dfs[channel]
        
        coords_target = get_coords(main_channel_df,x_name='x0_global',y_name='y0_global',keep_nans = False)
        coords_to_tranform = get_coords(channel_to_link_df,x_name='x0_global',y_name='y0_global',keep_nans = False)
        
        affine = get_affine(coords_to_tranform, coords_target)
        
        dict_affine_transforms[f'{channel}->{main_channel_name}'] = affine
        
    return dict_affine_transforms
    

def plot_affine_transformation(dict_PSF_dfs, dict_affine_transforms, target_channel_name, source_channel_name, img_data=False):
    """
    Plot affine transformation between two channels and visualize the results.

    Parameters
    ----------
    dict_PSF_dfs : dict of DataFrame
        A dictionary where keys are channel names and values are DataFrames
        containing the coordinates for each channel.
        
    dict_affine_transforms : dict 
        A dictionary containing the estimated affine transformations for each channel,
        with keys formatted as 'channelX->channelY' indicating the source and target channels.

    target_channel_name : str
        The name of the channel to which the source channel will be transformed.

    source_channel_name : str
        The name of the channel that will be transformed.

    img_data : array_like, optional
        An optional image array to be displayed in the background of the plot. 
        If provided, the image will be shown behind the scatter plots. Default is False.

    Returns
    -------
    None
    """
    # Retrieve DataFrames for the specified channels
    df_target_channel = dict_PSF_dfs[target_channel_name]
    df_source_channel = dict_PSF_dfs[source_channel_name]

    # Get coordinates for both channels
    target_coordinates = get_coords(df_target_channel, x_name='x0_global', y_name='y0_global', keep_nans=False)
    source_coordinates = get_coords(df_source_channel, x_name='x0_global', y_name='y0_global', keep_nans=False)

    # Compute the affine transformation

    
    # Apply the transformation to the source channel's coordinates
    transformed_coordinates = dict_affine_transforms[f'{source_channel_name}->{target_channel_name}'](source_coordinates)

    # Plot the results
    plt.figure(figsize=(8, 6))
    plt.scatter(target_coordinates[:, 1], target_coordinates[:, 0], marker='x', label='Reference (Target)',c='r')
    plt.scatter(source_coordinates[:, 1], source_coordinates[:, 0], marker='x', label='Initial (Source)',c='g')
    plt.scatter(transformed_coordinates[:, 1], transformed_coordinates[:, 0], marker='x', label='Transformed',c='y')
    plt.title(f'Affine Transformation: {source_channel_name} to {target_channel_name}')
    plt.xlabel('X Coordinates')
    plt.ylabel('Y Coordinates')
    plt.legend()
    
    if img_data is not None:
        plt.imshow(img_data)
    
    plt.show()


#%%Processing of full channel data 

def validate_limits_dict(all_channels_limits_dict, N_channels):
    """
    Validate that the provided limits dictionary matches the expected structure.

    Parameters
    ----------
    all_channels_limits_dict : dict
        A dictionary containing limits for selecting PSFs for each channel. 
        Keys should be in the format 'channel{index}' (e.g., 'channel0', 'channel1').
    
    N_channels : int
        The expected number of channels, used to determine the expected keys in the limits dictionary.

    Raises
    ------
    ValueError
        If the provided limits dictionary keys do not match the expected keys based on the number of channels.
    """
    expected_keys = {f'channel{idx}' for idx in range(N_channels)}
    provided_keys = set(all_channels_limits_dict.keys())
    
    if provided_keys != expected_keys:
        raise ValueError(f"Provided limits dictionary keys ({provided_keys}) do not match the expected keys ({expected_keys}).")

def extract_psf_data_for_all_channels(binary_PSF_data,image_data,PSF_windowsize):
    
    """
   Extract and fit PSF data for all channels from binary PSF data and corresponding image data.

   Parameters
   ----------
   binary_PSF_data : numpy.ndarray
       A 3D array containing binary PSF data for each channel.
   image_data : numpy.ndarray
       A 3D array containing image data for each channel corresponding to the binary PSF data.
   PSF_window_size : int
       The size of the window used for extracting PSF data from the image.


   Returns
   -------
   dict
       A dictionary where each key corresponds to a channel, and the value is a DataFrame containing the extracted and fitted PSF data for that channel.
   """
    
    #add option to do gauss or not 

    N_channels = binary_PSF_data.shape[0]
    
    if image_data.shape[0] != N_channels:
       raise ValueError(f"Number of channels in image_data ({image_data.shape[0]}) does not match "
                        f"the number of channels in binary_PSF_data ({N_channels}).")
   

    channels_dict = {}
    
    for channel_idx in range(N_channels):
        print(channel_idx)
        df = extract_and_fit_psf_data(binary_PSF_data[channel_idx,...],
                                      image_data[channel_idx,...],
                                      PSF_windowsize)
        
            
        channels_dict[f'channel{channel_idx}'] = df
        
    return channels_dict


def filter_dict_PSF_dfs(dict_PSF_dfs,limits_dict): 
    
    N_channels = len(dict_PSF_dfs)

    # Validate the limits dictionary
    validate_limits_dict(limits_dict, N_channels)
    
    dict_PSF_dfs_filtered = {}
    
    for channel_idx in range(N_channels):
        
        df = dict_PSF_dfs[f'channel{channel_idx}']

            
        limits_dict_channel = limits_dict[f'channel{channel_idx}']
        
        df = select_PSFs_from_df(df, limits_dict_channel)
            
        dict_PSF_dfs_filtered[f'channel{channel_idx}'] = df
        
        
    return dict_PSF_dfs_filtered
        
    
    
    
        
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
    angles = np.arctan2(y,x)
    
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
    
def get_coords(df,x_name = 'centroid-1',y_name = 'centroid-0',keep_nans = False):
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
    coords = np.array([df[y_name],df[x_name]]).T
    
    if not keep_nans:
        coords = coords[~np.isnan(coords).any(axis=1)]
    
    return coords
    


def show_PSFs_channel_old(data,coords,channel_num):
    """
    
    This function visualizes the specified channel by displaying the
    image corresponding to that channel. It also overlays the coordinates of detected PSFs.
    
    Parameters
    ----------
    data : numpy.ndarray
        3D array containing the PSF data across multiple channels. The shape should be (channels, height, width).
    
    coords : numpy.ndarray
        2D array of shape (n, 2) containing the coordinates of the detected PSFs, where `n` is the number
        of detected PSFs. Each row corresponds to [y_coordinate, x_coordinate].
    
    channel_num : int
        Index of the channel to be displayed from the `data` array.
    
    Returns
    -------
    None
        This function does not return any values. It generates a plot directly.
    """

    
    plt.imshow(data[channel_num,...])
    plt.scatter(coords[:,1],coords[:,0],marker='x',color='r')
    
    
    
def show_PSFs_channel(data,
                      dict_PSF_dfs,
                      channel_num,
                      x_name = 'x0_global',
                      y_name = 'y0_global'):
    """
    
    """
    df = dict_PSF_dfs[f'channel{channel_num}']
    coords = get_coords(df,x_name,y_name)

    
    plt.imshow(data[channel_num,...])
    plt.scatter(coords[:,1],coords[:,0],marker='x',color='r')

#%%

def point_union(list_a_in,list_b_in,min_dist):
    
    list_a = np.array(list_a_in)
    list_b = np.array(list_b_in)
    
    dist_M = distance_matrix(list_a, list_b)
    idx = dist_M.min(axis=0)>min_dist
    
    return np.concatenate((list_a, list_b[idx]))







