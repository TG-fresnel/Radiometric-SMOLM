# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 16:40:52 2025

@author: Storm4
This code reads a pupil image and gives back its properties when binarized
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from skimage.measure import label, regionprops_table
from skimage.morphology import (
    remove_small_objects,
    remove_small_holes,
    binary_dilation,
    binary_erosion,
    square,
    convex_hull_object
)


fontsize = 15
plt.rcParams["font.size"] = str(fontsize)



def search_QE(the_wavelength, camera_type='ORCA'):
    
    wavelength = np.array([400,550,700,800])
    
    if camera_type == 'ORCA':
        QE = np.array([0.65, 0.80, 0.70, 0.50])
    
    elif camera_type == 'ORCA BT':
        QE = np.array([0.72, 0.95, 0.83, 0.58])
        
    else :
        print("No such calera type. Options are ORCA and ORCA BT.")
        print("Using ORCA values as default")
        QE = np.array([0.65, 0.80, 0.70, 0.50])
    

    
    idx = (np.abs(wavelength - the_wavelength)).argmin()
    
    return QE[idx]

def binarize(image, threshold=1, radius=1, smin=1000):
    """
    The following function binarizes the image and then eliminates
    small holes and/or small objects.

    Parameters
    ----------
        threshold (float): multiplicative factor with respect the mean of the array
        radius (float): radius of the erosion and dilation steps
        smin (int):  minimum elements for a hole or an object to not be removed

    Return
    ------
    mask (array-like): A binary image of the ROIs.
    """

    # First we threshold the image with respect its mean and the given factor
    mask = image > (threshold * np.mean(image))
    # Erode to eliminatte noise
    mask = binary_erosion(mask, square(radius))
    # Dilate back
    mask = binary_dilation(mask, square(radius))
    # Remove small holes within the windows
    mask = remove_small_holes(mask, smin)
    # Remove small objects within the windows
    mask = remove_small_objects(mask, smin)
    return mask



img_path = "C:/Users/manip chido/Desktop/HexBFP/20250303/Pupil/"
file = 'BFP00008.tif'

threshold = 0.8
wavelength = 680

convert_to_photons = True
flip_flag = True #ORCA camera flips the image automatically, which for the pupil is useful


#Read pupill image
img = plt.imread(img_path+file)

#Undo flipping if needed
if flip_flag == False:
    #Undo the default flip
    img = np.flip( np.flip(img, 0), 1)
    
#Convert to photons if wanted  
if convert_to_photons==True:
    QE = search_QE(550, camera_type='ORCA BT')
    conversion = 0.24 #electrons/counts
    offset = 100.
    img = conversion*(img-offset)/QE
    img  = img*(img>0)
    
    

# Binarize to define the areas of interest
mask = binarize(img, threshold=threshold, radius=3, smin=1000)
mask = convex_hull_object(mask)
mask = label(mask)
properties = (
    "centroid",
    "orientation",
    "axis_major_length",
    "axis_minor_length",
    "bbox",
)
props_dict = regionprops_table(mask, properties=properties)
props = pd.DataFrame.from_dict(props_dict)

center = (np.round(props["centroid-0"][0]),np.round(props["centroid-1"][0]))
print("Computed centroid: ", center)
print("Computed axes: ", props["axis_major_length"][0]," ", props["axis_minor_length"][0])

data = img*mask

plt.figure()
plt.imshow(mask, cmap='inferno')
plt.scatter(center[1] , center[0],label=str(center), marker='+', color='green')
plt.xlabel(r"$x$ [pix]")
plt.ylabel(r"$y$ [pix]")
plt.title("Binary pupil")
plt.legend()
plt.tight_layout()
# plt.savefig(analysis_path+"FullPupil.svg", transparent=True)
plt.show()

plt.figure()
plt.imshow(data, cmap='inferno')
plt.scatter(center[1] , center[0],label=str(center), marker='+', color='green')
plt.xlabel(r"$x$ [pix]")
plt.ylabel(r"$y$ [pix]")
plt.title("Pupil")
plt.legend()
plt.tight_layout()
# plt.savefig(analysis_path+"FullPupil.svg", transparent=True)
plt.show()
