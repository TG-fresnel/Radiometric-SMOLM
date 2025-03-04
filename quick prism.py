# -*- coding: utf-8 -*-
"""
Created on Tue Mar  4 16:19:27 2025

@author: Tobias

adapdet version of quick pupil for 6 channels
"""



import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


from rSMOLM_module import *
from utility_functions import *


fontsize = 15
plt.rcParams["font.size"] = str(fontsize)

#%%

def plot_radial_lines(target_x,target_y,num_radial_sections,length_radial_lines = 800):
    
    alpha_deg = 360/num_radial_sections #angle between the angualr lines, for 5 outer section is 360/5
    alpha_rad = np.deg2rad(alpha_deg)
    
    for i in range(0,num_radial_sections + 1):
        
        
        angle = alpha_rad * i
    

        x1 = target_x
        y1 = target_y
        
        x2 = x1 + np.cos(angle)*length_radial_lines
        y2 = y1 + np.sin(angle)*length_radial_lines
        
        print(i)
        
        plt.plot([y1, y2],[x1, x2],'r',lw=1,ls = '--')
    
#%%


img_path = 'C:/Users/Tobias/Documents/Data/HexBFP/20250303/Pupil/'
file = 'BFP00008.tif'

#target values of the prism center
target_x = float(input("input target x position "))
target_y = float(input("input target y position "))



num_radial_sections = 5 


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

#%%
plt.figure()
plt.imshow(data, cmap='inferno')
plt.colorbar()
plt.scatter(center[1] , center[0],label='current center', marker='+', color='green')
plt.scatter(target_y , target_x,label='target center', marker='+', color='red')
plot_radial_lines(target_x,target_y,num_radial_sections)
plt.xlabel(r"$x$ [pix]")
plt.ylabel(r"$y$ [pix]")
plt.title("Pupil")
plt.legend()
plt.tight_layout()

# plt.savefig(analysis_path+"FullPupil.svg", transparent=True)
plt.show()
