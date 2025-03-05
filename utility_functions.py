# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 13:45:15 2024

@author: Tobias

Last edit : Dec 19 2024, Adren Casseville
"""
import matplotlib.pyplot as plt
import skimage as ski
import numpy as np


# %% functions related to loading image and extracting channels
def load_img(img_path, frame_nb_axis):
    """
    loads a img and return the sum of the frames in it.

    Parameters
    ----------
        img_path (str): The path to the image.
        frame_nb_axis (int): The axis along which the sum of images is made.

    Return
    ------
    img (array-like): A 2D array of the data.
    """
    raw_data = ski.io.imread(img_path)
    shape = raw_data.shape

    if len(shape) == 3:
        img = np.sum(raw_data, axis=frame_nb_axis)
    elif len(shape) == 2:
        img = ski.io.imread(img_path)
    else:
        print("Error with the shape of the data")
        return None
    return img


def load_data(data_path, frame_nb_axis=0):
    """
    loads the data and sums everything to return a single img.

    Parameters
    ----------
        data_path (str): The path to the data.
        frame_nb_axis (int): The axis along which the sum of images is made.

    Return
    ------
    img (array-like): A 2D array of the data.
    """
    if isinstance(data_path, list):
        img = load_img(data_path[0], frame_nb_axis)
        for i in range(1, len(data_path)):
            img += load_img(data_path[i], frame_nb_axis)
    else:
        img = load_img(data_path, frame_nb_axis)

    # Invert the image
    #New ORCA Camera flips x and y automatically, so we undo this
    img = np.flip( np.flip(1.0 * img, 0), 1)
    
    return img


def load_channels(img_in, ROI_props, extra_box_pixels=0):
    """
    Loads each channels data from image based on ROI_props

    Parameters
    ----------
    img_in (2d array): 2D input image
    ROI_props (dict): dict that contains 'centroid-0','centroid-1','axis_major_length'.
    extra_box_pixels (int, optional): extra margin pixels when extracting channels. The default is 0.

    Returns
    -------
    data (3d-array): 3d array containing the data of all channels, dimension 0 corresponds to number of channels,
        dimensions 1 and 2 are spatial.

    """
    N_channels = len(ROI_props)

    # rounded integers of the center coordinates of each channel
    ROI_centers = np.array([ROI_props["centroid-0"], ROI_props["centroid-1"]])

    # Make uniform box length choosing the largest
    box_length = (
        np.round(np.max([ROI_props["axis_major_length"]])).astype(int)
        + extra_box_pixels
    )

    data = extract_channels(img_in, N_channels, box_length, ROI_centers)

    return data


def extract_channels(img, N_channels, box_length, ROI_centers):
    """
    Extracts each channels data from image

    Parameters
    ----------
    img (2D array): 2D input image.
    N_channels (int): number of channels.
    box_length (int): box length of each channels subimages.
    ROI_centers (2d array): array containing the centers of each channel, dimensions of array 2 x N_channels.

    Returns
    -------
    data (3D-array): 3d array containing the data of all channels, dimension 0 corresponds to number of channels,
        dimensions 1 and 2 are spatial.

    """
    data = np.zeros([N_channels, box_length, box_length])
    w = box_length // 2
    for channel_idx in range(N_channels):
        y, x = ROI_centers[:, channel_idx]
        x = round(x)
        y = round(y)
        data[channel_idx, :, :] = img[
            y - w : y + w + 1,
            x - w : x + w + 1,
        ]

    return data


def extract_masks(ROI_masks, ROI_props, extra_box_pixels=0):
    """

    Parameters
    ----------
    ROI_masks (2d-integer array): Mask array where each channels mask is indicated by an integer.
    ROI_props (dict): dict that contains: 'centroid-0','centroid-1','axis_major_length'.
    extra_box_pixels (int, optional): extra margin pixels. The default is 0.

    Returns
    -------
    extracted_masks (3D-boolean array)
    """

    N_channels = len(ROI_props)

    # rounded integers of the center coordinates of each channel
    ROI_centers = np.array([ROI_props["centroid-0"], ROI_props["centroid-1"]])

    # Make uniform box length choosing the largest
    box_length = (
        np.round(np.max([ROI_props["axis_major_length"]])).astype(int)
        + extra_box_pixels
    )

    extracted_masks = extract_channels(
        ROI_masks, N_channels, box_length, ROI_centers
    )

    for i in range(N_channels):
        extracted_masks[i, ...] = extracted_masks[i, ...] == (i + 1)

    return extracted_masks.astype(bool)


def get_limits(shape, limits_x, limits_y):
    x1, x2 = limits_x
    y1, y2 = limits_y
    if x1 is None:
        x1 = 0
    if x2 is None:
        x2 = shape[2]
    if y1 is None:
        y1 = 0
    if y2 is None:
        y2 = shape[1]
    return x1, x2, y1, y2


def find_max(img, min_intensity, return_value=False):
    value = np.max(img)
    if value > min_intensity:
        coord = np.where(img == value)
        if return_value:
            return [value, coord[1][0], coord[0][0]]
        return [coord[1][0], coord[0][0]]
    return None


def display_channels(img, data_psf):
    fig, ax = plt.subplots(1, len(img))
    for i in range(len(img)):
        ax[i].imshow(img[i])
        ax[i].set_title(f"Detected PSFs in channel {i}")
        ax[i].axis("off")
        for j in range(len(data_psf[i])):
            x, y = data_psf[i][j]
            ax[i].plot(
                x,
                y,
                marker="x",
                color="r",
                markersize=1.5,
            )
            ax[i].text(
                x + 3,
                y,
                str(j),
                color="white",
                fontsize=5,
            )
    plt.tight_layout()
    plt.show()


def display_channel(img, coords):
    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.set_title(f"Detected PSFs in channel")
    ax.axis("off")
    for i in range(len(coords)):
        x, y = coords[i]
        ax.plot(
            x,
            y,
            marker="x",
            color="r",
            markersize=1.5,
        )
        ax.text(
            x + 3,
            y,
            str(i),
            color="white",
            fontsize=5,
        )
    plt.show()


def search_QE(the_wavelength, camera_type="ORCA"):

    wavelength = np.array([400, 550, 700, 800])

    if camera_type == "ORCA":
        QE = np.array([0.65, 0.80, 0.70, 0.50])

    elif camera_type == "ORCA BT":
        QE = np.array([0.72, 0.95, 0.83, 0.58])

    else:
        print("No such calera type. Options are ORCA and ORCA BT.")
        print("Using ORCA values as default")
        QE = np.array([0.65, 0.80, 0.70, 0.50])

    idx = (np.abs(wavelength - the_wavelength)).argmin()

    return QE[idx]


def counts_to_photons(img_data):
    """
    Converts the value of pixels from counts to photons
    """
    # Transform to photons
    QE = search_QE(700, camera_type="ORCA BT")
    conversion = 0.24  # electrons/counts
    offset = 100.0

    # Transform data into photons
    img_data = conversion * (img_data - offset) / QE
    return img_data * (img_data > 0)  # Clip
