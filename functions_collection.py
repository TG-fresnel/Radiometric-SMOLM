import numpy as np 

# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 15:40:54 2024

@author: Tobias
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
from matplotlib.patches import ConnectionPatch,Rectangle
from matplotlib.collections import PatchCollection
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
from skimage.morphology import erosion,disk, area_opening
from skimage.measure import regionprops_table,label,centroid 
from scipy.signal import convolve2d
from scipy.spatial import distance_matrix
from scipy.ndimage import distance_transform_edt
from skimage.transform import AffineTransform
from skimage.measure import ransac
from skimage.filters import threshold_otsu
from skimage.segmentation import watershed
from skimage.draw import ellipse_perimeter
from scipy.optimize import curve_fit
from tqdm import tqdm
import warnings

# Personal modules


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


def Vflip(array):
    """ Flip the image vertically.
    """
    return np.flip(array, 0)

def frame_mask(length):
    '''creates squared mask which is 0 everywhere and 1 at the borders'''
    if length < 3:
        print('too small length for frame mask')
        return
    
    total = length*4-4
    conv_mask = np.zeros((length,length))
    conv_mask[0,:] = 1/total
    conv_mask[-1,:] = 1/total
    conv_mask[:,0] = 1/total
    conv_mask[:,-1] = 1/total

    return conv_mask


def disk(radius=2, form=[4, 4]):
    """Generates a binary disk mask.

   Arguments:
       radius (int): The radius of the disk (default: 2).
       form (list): The shape of the mask in the form [height, width] (default: [4, 4]).
   """
    y, x = np.ogrid[-(form[0] - 1) / 2: (form[0] - 1) / 2 + 1,
           -(form[1] - 1) / 2: (form[1] - 1) / 2 + 1]
    mask = (x) ** 2 + (y) ** 2 <= radius ** 2
    return mask



def ellips_perimeter_mask(maj_ax,min_ax,orientation=0):
    
   
    
    img = np.zeros((maj_ax*2+1, maj_ax*2+1), dtype=np.uint8)
    rr, cc = ellipse_perimeter(maj_ax, maj_ax,maj_ax,min_ax,orientation,shape=img.shape)
    img[rr, cc] = 1

    return img[min(rr):max(rr)+1,min(cc):max(cc)+1]/img.sum()

def conv_psf_finder(img,masks,plot_all = False,plot_final = False):
    convolved_img_stack = np.zeros((*img.shape,len(masks)+1))
    convolved_img_stack[...,0] = img
    
    for i,mask in enumerate(masks):
           convolved_img_stack[...,i+1] =  convolve2d(img,mask,mode='same')
           
    stack_diff = np.diff(convolved_img_stack, axis=2)
    final = np.all(stack_diff< 0, axis=2)
    #compact way to check each convolved array is smaller thasn the previous one 

    if plot_all: 
        
        fig1, axes1 = plt.subplots(3,len(masks),layout='constrained',figsize = (3*len(masks),3*3))
        
        for i,mask in enumerate(masks):
               
            axes1[0,i].imshow(mask)
            axes1[1,i].imshow(convolved_img_stack[...,i+1])
            axes1[2,i].imshow( np.all(stack_diff[...,:i+1]< 0, axis=2))
            
    if plot_final:
        
        fig2, axes2 = plt.subplots(1,1,layout='constrained',figsize = (6,6))

        axes2.imshow(img,cmap='gray')
        axes2.contour(final,alpha=0.5,colors = ['red'],linewidths=2)

    return final 



def get_erroded_masks(ROI_masks,roi_props,footprint_size,extra_box_pixels= 10):

    centroids_roi = np.round(np.array([roi_props['centroid-0'],roi_props['centroid-1']])).astype(int)

    #Make uniform box length choosing the largest
    box_length = ( int(np.round(np.max([roi_props['axis_major_length']]))) 
                  + extra_box_pixels)
    

    erosion_mask = disk(footprint_size)
    erroded_masks = np.zeros((ROI_masks.max(),box_length,box_length))
    for channel_idx in range(ROI_masks.max()):
        
        channel_mask_full_frame = ROI_masks==(channel_idx+1)
        channel_mask_zoom = take_from(channel_mask_full_frame,
                                     box_length,[centroids_roi[0][channel_idx],centroids_roi[1][channel_idx]])*disk(box_length//2, [box_length, box_length])
                        
        erroded_masks[channel_idx,...] = erosion(channel_mask_zoom,erosion_mask)

    
    return erroded_masks

def extract_psfs(img,centroids,windowsize):
    
    psfs_stack = np.zeros((len(centroids),windowsize,windowsize))
    
    for i,coords in enumerate(centroids):
        
        psfs_stack[i,...] = take_from(img, windowsize, [int(coords[0]),int(coords[1])])
        
    return psfs_stack

def gaussian_2d(x, y, amp, xo, yo, sigma_x, sigma_y, theta, offset):
    x0 = float(xo)
    y0 = float(yo)
    a = (np.cos(theta)**2) / (2 * sigma_x**2) + (np.sin(theta)**2) / (2 * sigma_y**2)
    b = -(np.sin(2*theta)) / (4 * sigma_x**2) + (np.sin(2*theta)) / (4 * sigma_y**2)
    c = (np.sin(theta)**2) / (2 * sigma_x**2) + (np.cos(theta)**2) / (2 * sigma_y**2)
    return offset + amp * np.exp(-(a * ((x - x0)**2) + 2 * b * (x - x0) * (y - y0) + c * ((y - y0)**2)))   

def watershed_psf(binary,return_mask=False):
    
    dist = distance_transform_edt(binary)
    watershed_res = watershed(-dist,mask =binary )
    
    if return_mask:
        return watershed_res
    
    return watershed_res.max()
    
def get_centroid_list(df):
    
    return list(zip(df['centroid-1'],df['centroid-0']))

def get_gauss_pos_list(df):
    
    
    positions_1 = np.array(list(zip(df['gauss_y0_global'],df['gauss_x0_global'])))
      
    #for psfs where the gauss-fit failed the centroid is considered
    nan_idx_1 = np.isnan(positions_1)
   

    
    if nan_idx_1.sum() > 0:
        print(f'{ (nan_idx_1.sum() )/2 } psfs-postions have no gauss-fit, centroid is used instead')
        
        centroids_1 = np.array(list(zip(df['centroid-0'],df['centroid-1'])))
        
        
        positions_1[nan_idx_1] = centroids_1[nan_idx_1]
     
    
    return positions_1

def show_psfs_FOV_from_list(channels,centroids,nrow=2,ncols=2,psf_window_size=9):
    
    '''this function takes the stacekd images of the channel as a first argument and a list of of centroids as a second. 
    It generates a plot that indicates the locations of the psfs'''
    
    '''outdated'''
    
    fig, axes  = plt.subplots(nrow,ncols,figsize=(7,7),sharex=True,sharey=True)
    
    for i,(channel,centroid) in enumerate(zip(channels,centroids)):
        axes.flatten()[i].imshow(channel)
        ceny,cenx =  zip(*centroid)
        axes.flatten()[i].scatter(cenx,ceny,color='black')
        
        rectangeles = []
        for x,y  in zip(cenx,ceny):
            axes.flatten()[i].add_patch(Rectangle((x-psf_window_size/2,y-psf_window_size/2), psf_window_size, psf_window_size,linewidth=1, edgecolor='red', facecolor='none'))


def fit_2d_gaussian(data,theta_guess = 0,
                    sigma_x_guess = None,
                    sigma_y_guess = None,
                    apply_bounds = True,
                    bounds =([0,0,0,0,0,-np.pi,0],
                             [np.inf,np.inf,np.inf,np.inf,np.inf,np.pi,np.inf]) ):
    
    size_x, size_y = data.shape
    x = np.linspace(0, size_x - 1, size_x)
    y = np.linspace(0, size_y - 1, size_y)
    x, y = np.meshgrid(x, y)

    # Flatten the data and coordinate grids for curve fitting
    xdata = np.vstack((x.ravel(), y.ravel()))
    ydata = data.ravel()

    if not sigma_x_guess:
        sigma_x_guess =  size_x / 4
        
    if not sigma_y_guess:
        sigma_y_guess =  size_y / 4

    # Initial guess for the parameters: amp, xo, yo, sigma_x, sigma_y, theta, offset
    initial_guess = (np.max(data), size_x / 2, sigma_x_guess, sigma_x_guess, size_y / 4, theta_guess, np.min(data))

    # Perform the curve fitting
    if apply_bounds:
        params, cov = curve_fit(lambda xy, amp, xo, yo, sigma_x, sigma_y, theta, offset: 
                              gaussian_2d(xy[0], xy[1], amp, xo, yo, sigma_x, sigma_y, theta, offset), 
                              xdata, ydata,
                              p0=initial_guess,
                              bounds = bounds)

    else: 
        params, cov = curve_fit(lambda xy, amp, xo, yo, sigma_x, sigma_y, theta, offset: 
                              gaussian_2d(xy[0], xy[1], amp, xo, yo, sigma_x, sigma_y, theta, offset), 
                              xdata, ydata,
                              p0=initial_guess)
        
            
    # Extract the fitted parameters
    amp, xo, yo, sigma_x, sigma_y, theta, offset = params

    # Create fitted data using the fitted parameters
    fitted_data = gaussian_2d(x, y, amp, xo, yo, sigma_x, sigma_y, theta, offset)

    return params,cov, fitted_data

        
        

def create_rectangles_vectorized(coordinates_x,coordinates_y, widths, heights):
    '''function that allows to create all rectangle objects in an efficient way from list'''
    
    coordinates = list(zip(coordinates_x,coordinates_y))
    rect_params = np.hstack([coordinates, widths[:, None], heights[:, None]])
    rects = np.apply_along_axis(lambda p: Rectangle((p[0]-p[2]/2, p[1] - p[3]/2), p[2], p[3],linewidth=1, edgecolor='red', facecolor='none'), 1, rect_params)
    return rects    
        
def show_psfs_from_df(data, df,show_psfs=False,show_matches = False,show_gauss=False,nrow=2,ncols=2,psf_window_size=9,columns_ls = None):
      
    '''this function takes the stacekd images of the channel as a first argument and a df containing all channels. 
    It generates a plot that indicates the locations of the psfs'''
    
    #plt.switch_backend('QtAgg')
    
    fig, axes  = plt.subplots(nrow,ncols,figsize=(7,7),sharex=True,sharey=True)
    
    if show_matches:
        connec1 = []
        connec2 = []
        connec3 = []

        
    
    if show_psfs:

        #creates new window to show indormation about selected psf
        fig_psfs, ax_psfs = plt.subplot_mosaic([['a)', 'b)'], ['a)', 'c)',], ['d)', 'd)'], ['d)', 'd)']],
                              height_ratios = [1,1,1,1],width_ratios = [1,1],figsize=(5,7))

        #fig.canvas.manager.window.move(100,100)
        #fig_psfs.canvas.manager.window.move(850,100)    
        
        ax_psfs['b)'].set_xticks([])
        ax_psfs['b)'].set_yticks([])
        
        ax_psfs['c)'].set_xticks([])
        ax_psfs['c)'].set_yticks([])
        
        #variables to store connections so they can be removed

       
    
    if show_gauss and 'gauss_amp' in df.columns:
    
                
            fig_gaus, ax_gaus = plt.subplots(nrow,ncols,figsize=(7,7))
            
            #fig.canvas.manager.window.move(100,100)
            #fig_gaus.canvas.manager.window.move(1300,100)    
            
            
            size_x, size_y = df.iloc[0]['psfs'].shape
            x = np.linspace(0, size_x - 1, size_x)
            y = np.linspace(0, size_y - 1, size_y)
            xv, yv = np.meshgrid(x, y)
            
            
            


  
    
    channel_names = df.index.levels[0]
    rects = {}
    rects_plot_ls = []
    marker_plot_ls = []
    #get centroid position and create rectangles to indicate each psf
    for i,channel_name in enumerate(channel_names):
        ceny = df.loc[channel_name]['centroid-0']
        cenx = df.loc[channel_name]['centroid-1']
        widths = np.ones(len(cenx))*psf_window_size
        lengths = np.ones(len(cenx))*psf_window_size
        
        rects = create_rectangles_vectorized(cenx,ceny,widths,lengths)
        rects_collection = PatchCollection(rects, match_original=True)
        
        rect_plot = axes.flatten()[i].add_collection(rects_collection)       
        markers, = axes.flatten()[i].plot(cenx,ceny,color='orange',ls='',marker = 'o', markersize=3 ,picker = True,pickradius = 5)
        axes.flatten()[i].imshow(data[i,...])
        rects_plot_ls.append(rect_plot)
        marker_plot_ls.append(markers)
       
    
    def draw_connection_from_row(psf_row,channels_to_link,ax_nr):
       
        for i,channel in enumerate(channels_to_link):
            colum_name = channel + '_closest_point_idx'
            matched_psfs_idx = int(psf_row.loc[colum_name])
            
        
            active_y = psf_row['centroid-0']
            active_x = psf_row['centroid-1']
        
            linked_y = df.loc[channel,'centroid-0'].iloc[matched_psfs_idx]
            linked_x = df.loc[channel,'centroid-1'].iloc[matched_psfs_idx]
            
            linked_ax_idx = int(channel[-1])
            active_axis = axes.flatten()[ax_nr]
            linked_axis = axes.flatten()[linked_ax_idx]
            
            con = ConnectionPatch(xyA=(active_x,active_y),
                                  xyB=(linked_x,linked_y),
                                  axesA=active_axis,
                                  axesB=linked_axis,
                                  coordsA='data',
                                  coordsB='data',
                                  color = 'tab:cyan')
            
            #the connectino patch needs to be plotted on the axes witht the bigger index to be seen in all subplots
            if linked_ax_idx > ax_nr:
                connection = linked_axis.add_artist(con)
            else:
                connection = active_axis.add_artist(con)
                
            if i == 0:  connec1.append(connection)
            if i == 1:  connec2.append(connection)
            if i == 2:  connec3.append(connection)
                

    def plot_matches(psf_row,ax_nr):

        active_channel = f'channel{ax_nr}'
        channels_to_link = list(channel_names)
        channels_to_link.remove(active_channel)
        
        draw_connection_from_row(psf_row,channels_to_link,ax_nr)

       
        fig.canvas.draw()
        
        
    def on_click(event):
        '''function that prints information about the selected psf and id plot_psf true shows it in the second figure'''

        clicked_ax = np.array( [event.artist.axes == ax for ax in axes.flatten()])
        ax_nr = np.where(clicked_ax)[0][0]
        thisline = event.artist
        xdata = thisline.get_xdata()
        ydata = thisline.get_ydata()
        ind = event.ind
        out = df.loc[f'channel{ax_nr}'].iloc[ind[0]]
        psf = df.loc[f'channel{ax_nr}','psfs'].iloc[ind[0]]
       
        if columns_ls:
            print(out[columns_ls])
        else:
            print(out)
            
        if show_psfs:
            ax_psfs['a)'].clear()
            ax_psfs['d)'].clear()
            binary = psf>out.loc['threshold']
            watershed_img  = watershed_psf(binary,return_mask=True)
            windowsize = psf.shape[0]
            ceny = out['centroid-0']
            cenx = out['centroid-1']
            k = windowsize//2
            n =-windowsize%2
    
            ceny_local = ceny-(int(ceny) - k + 1 - n)
            cenx_local = cenx-(int(cenx) - k + 1 - n)
            ax_psfs['a)'].imshow(psf)
            ax_psfs['a)'].plot(cenx_local,ceny_local,marker = 'x',c = 'r',ls='')
            ax_psfs['a)'].plot(centroid(psf)[1],centroid(psf)[0],marker = 'x',c = 'purple',ls='')
            ax_psfs['b)'].imshow(binary)
            ax_psfs['c)'].imshow(watershed_img)
            ax_psfs['d)'].hist(psf.flatten())
            ax_psfs['d)'].axvline(x = out.loc['threshold'],c='r',ls='--')

            
        
            
            fig_psfs.suptitle(f'channel{ax_nr}   psf_idx:{ind[0]}')
            fig_psfs.canvas.draw()

        if show_matches:
            plot_matches(out,ax_nr)
            
        if show_gauss and 'gauss_amp' in df.columns:
                       
            max_x_index = int(psf_window_size/2)
            max_y_index = int(psf_window_size/2)

            
            
            gauss_fit = gaussian_2d(xv,
                                   yv,
                                   out['gauss_amp'],
                                   out['gauss_x0'],
                                   out['gauss_y0'],
                                   out['gauss_sigma_x'],
                                   out['gauss_sigma_y'],
                                   out['gauss_theta'],
                                   out['gauss_offset'])
            
            psf_x_line_plot = psf[max_x_index,:]
            fit_x_line_plot = gauss_fit[max_x_index,:]
            
            psf_y_line_plot = psf[:,max_y_index]
            fit_y_line_plot = gauss_fit[:,max_y_index]
            
            vmin = min(psf.min(),gauss_fit.min())
            vmax = max(psf.max(),gauss_fit.max())
            
            ax_gaus[0,0].clear()
            ax_gaus[0,1].clear()
            ax_gaus[1,0].clear()
            ax_gaus[1,1].clear()
            
            ax_gaus[0,0].imshow(psf,vmin=vmin,vmax=vmax)
            ax_gaus[0,1].imshow(gauss_fit,vmin=vmin,vmax=vmax)
            ax_gaus[0,1].plot(out['gauss_x0'],out['gauss_y0'],marker = 'o',color = 'red')
            ax_gaus[1,0].plot(psf_y_line_plot,c = 'tab:blue', ls = '-',label = 'data')
            ax_gaus[1,0].plot(fit_y_line_plot,c = 'tab:red', ls = '--',label = 'fit')
            ax_gaus[1,1].plot(psf_x_line_plot,c = 'tab:blue', ls = '-',label = 'data')
            ax_gaus[1,1].plot(fit_x_line_plot,c = 'tab:red', ls = '--',label = 'fit')

            ymin = min(psf.min(),gauss_fit.min())
            ymin = 0
            ymax = max(psf.max(),gauss_fit.max())

            ax_gaus[1,0].set_ylim((ymin,ymax))
            ax_gaus[1,1].set_ylim((ymin,ymax))
            ax_gaus[1,0].legend()
            ax_gaus[1,1].legend()
            
            
            fig_gaus.canvas.draw()

    def on_press(event):
        
        
        if event.key == 'e':
            for markers,rect_plot in zip(marker_plot_ls,rects_plot_ls):
                if markers.get_visible():
                    markers.set_visible(False)
                    rect_plot.set_visible(False)
                else:
                    markers.set_visible(True)
                    rect_plot.set_visible(True)
            fig.canvas.draw()
      
        if show_matches:
            if event.key == 'm':
                if len(connec1) > 0:
 
                    old_x_lim = axes.flatten()[0].get_xlim()
                    old_y_lim = axes.flatten()[0].get_ylim()
                    
                    connec1[-1].remove()
                    connec2[-1].remove()
                    connec3[-1].remove()
                    
                    del connec1[-1]
                    del connec2[-1]
                    del connec3[-1]
                         
                    fig.canvas.draw()

    def get_gaus_line_plot(event):
        
        

        if event.inaxes in ax_gaus.flatten()[:2]:
            
            
            if len(ax_gaus[0,0].lines) > 0:
                ax_gaus[0,0].get_lines()[1].remove()
                ax_gaus[0,0].get_lines()[0].remove()
                
            
            psf = ax_gaus[0,0].images[0].get_array()
            gauss_fit =ax_gaus[0,1].images[0].get_array()
            
            max_x_index = int(np.floor(event.xdata + 0.5))
            max_y_index = int(np.floor(event.ydata + 0.5))
            
            ax_gaus[0,0].axvline(x=max_x_index,c='r',ls='--')
            ax_gaus[0,0].axhline(y=max_y_index,c='r',ls='--')

            
            psf_x_line_plot = psf[max_x_index,:]
            fit_x_line_plot = gauss_fit[max_x_index,:]
            
            psf_y_line_plot = psf[:,max_y_index]
            fit_y_line_plot = gauss_fit[:,max_y_index]
                
            ax_gaus[1,0].clear()
            ax_gaus[1,1].clear()
              
            ax_gaus[1,0].plot(psf_y_line_plot,c = 'tab:blue', ls = '-',label = 'data')
            ax_gaus[1,0].plot(fit_y_line_plot,c = 'tab:red', ls = '--',label = 'fit')
            ax_gaus[1,1].plot(psf_x_line_plot,c = 'tab:blue', ls = '-',label = 'data')
            ax_gaus[1,1].plot(fit_x_line_plot,c = 'tab:red', ls = '--',label = 'fit')


            ymin = min(psf.min(),gauss_fit.min())
            ymin = 0
            ymax = max(psf.max(),gauss_fit.max())

            ax_gaus[1,0].set_ylim((ymin,ymax))
            ax_gaus[1,1].set_ylim((ymin,ymax))
            
            ax_gaus[1,0].legend()
            ax_gaus[1,1].legend()

            
            
            
        fig_gaus.canvas.draw()
            
            
             
             


                 
    if show_gauss and 'gauss_amp' in df.columns:
        
        gaus_line_plot = fig_gaus.canvas.mpl_connect('button_press_event', get_gaus_line_plot)
                   

    pick_event = fig.canvas.mpl_connect('pick_event', on_click)
    remove_connection = fig.canvas.mpl_connect('key_press_event', on_press)
                
             

    return fig 


    




def link_two_channels(df,chan1,chan2,position_method = 'centroid'):
    
    '''function that matches two channels by distances, 
    the closest point is referenced in a new column as well as  the distance to it'''
    
    df1 = df.loc[chan1]
    df2 = df.loc[chan2]

    if position_method == 'centroid':
        positions_1 = list(zip(df.loc[chan1]['centroid-0'],df.loc[chan1]['centroid-1']))
        positions_2 = list(zip(df.loc[chan2]['centroid-0'],df.loc[chan2]['centroid-1']))
        
    elif position_method == 'gauss_fit' and 'gauss_x0_global' in df.columns:
        
        positions_1 = np.array(list(zip(df.loc[chan1]['gauss_y0_global'],df.loc[chan1]['gauss_x0_global'])))
        positions_2 = np.array(list(zip(df.loc[chan2]['gauss_y0_global'],df.loc[chan2]['gauss_x0_global'])))
        
        #for psfs where the gauss-fit failed the centroid is considered
        
        nan_idx_1 = np.isnan(positions_1)
        nan_idx_2 = np.isnan(positions_2)


        
        print(f'{ (nan_idx_1.sum() + nan_idx_2.sum())/2 } psfs-postions have no gauss-fit, centroid is used instead')
        
        if nan_idx_1.sum() + nan_idx_2.sum() > 0:
            
            centroids_1 = np.array(list(zip(df.loc[chan1]['centroid-0'],df.loc[chan1]['centroid-1'])))
            centroids_2 = np.array(list(zip(df.loc[chan2]['centroid-0'],df.loc[chan2]['centroid-1'])))
            

            
            
            positions_1[nan_idx_1] = centroids_1[nan_idx_1]
            positions_2[nan_idx_2] = centroids_2[nan_idx_2]
            
            
        
    else:
        print('unknown position_method (centroid or gauss_fit) or gaus data not found')
        
    

    dist_M = distance_matrix(positions_1, positions_2)


    df.loc[chan1,f'{chan2}_closest_point_idx'] = dist_M.argmin(axis= 1)
    df.loc[chan1,f'{chan2}_closest_point_dist'] = dist_M.min(axis= 1)
    
    df.loc[chan2,f'{chan1}_closest_point_idx'] = dist_M.argmin(axis= 0)
    df.loc[chan2,f'{chan1}_closest_point_dist'] = dist_M.min(axis= 0)
    

    
def link_all_channels(df,position_method = 'centroid'):
    
    channel_names = df.index.levels[0]

    for i in range(len(channel_names)):
        
        chan_1 = channel_names[i]
        

        for j in range(len(channel_names[(i):])):
            
            chan_2 = channel_names[i+j]

            link_two_channels(df,chan_1,chan_2,position_method = position_method)
            print(chan_1,chan_2)
            if i==j:
                length = df.loc[chan_1].shape[0]
                df.loc[chan_1,f'{chan_1}_closest_point_idx'] = np.zeros(length)
                df.loc[chan_1,f'{chan_1}_closest_point_dist'] = np.zeros(length)
             
def energy_box_integration(df):
    
    psfs = np.array(df['psfs'].values.tolist())
    N_rows = len(psfs)
    psf_window_size = psfs[0].shape[0]

    border_pixels =np.concatenate( (psfs[:,0,:].reshape((N_rows,psf_window_size)),
                                    psfs[:,-1,:].reshape((N_rows,psf_window_size)),
                                    psfs[:,1:-1,0].reshape((N_rows,psf_window_size - 2)),
                                    psfs[:,1:-1,-1].reshape((N_rows,psf_window_size - 2))), axis = 1)
    
    border_pixels_mean = border_pixels.mean(axis = 1)
    energy = psfs.sum(axis=(1,2)) - border_pixels_mean*psf_window_size**2
    df['background_box_integration'] = list(border_pixels_mean)
    df['energy_box_integration'] = list(energy)
    
               
                
def get_E_splits(df,gauss = True,box_integration =True):
    
    channel_names = list( df.index.levels[0])

    for chan_1 in channel_names:
        

        link_channels = channel_names.copy()
        link_channels.remove(chan_1)

        for chan_2 in link_channels:
            
            
            
            chan2_match_idx = df.loc[chan_1,f'{chan_2}_closest_point_idx']
            
            if gauss:
            
                chan_1_energy = df.loc[chan_1,'total_int'].values
                chan_2_energy = df.loc[chan_2,'total_int'].iloc[chan2_match_idx].values
    
                df.loc[chan_1,f'{chan_2}_E_split_gauss'] = (chan_2_energy / chan_1_energy).tolist()
                
              
            if box_integration:
            
                chan_1_energy = df.loc[chan_1,'energy_box_integration'].values
                chan_2_energy = df.loc[chan_2,'energy_box_integration'].iloc[chan2_match_idx].values
    
                df.loc[chan_1,f'{chan_2}_E_split_box_integration'] = (chan_2_energy / chan_1_energy).tolist()


            
def matrix_plot_E_split(df,method = 'gauss',bins = 100):
    #this function needs to be improved !!
    if method not in ['gauss','box_integration']:
        
        print('method needs to be either gauss or box_integration ')
        
        return
        
        
    channel_names = list( df.index.levels[0])
    nr_channels = len(channel_names)
    fig,axes = plt.subplots(nr_channels,nr_channels)
    

    for i,chan_1 in enumerate(channel_names):
        

        link_channels = channel_names.copy()
        link_channels.remove(chan_1)

        for j,chan_2 in enumerate(link_channels):
            
            if j >= i: j=j+1
            axes[i,j].hist(df.loc[chan_1,f'{chan_2}_E_split_box_integration'],bins =bins )

def create_psfs_df(labeled,data,window_size,redo_centroids=False):
    
    dfs_tmp = []
    k = window_size//2
    n =-window_size%2
    
    for i in range(4):
        
        df  = pd.DataFrame(regionprops_table( labeled[i,...],data[i,...],properties=('label','intensity_mean','centroid','num_pixels','orientation')))
        

        centroids = list(zip(df['centroid-0'],df['centroid-1']))
        
        cenx = np.array(centroids)[:,0].astype(dtype=int)
        ceny = np.array(centroids)[:,1].astype(dtype=int)
        
        up = cenx - k + 1 - n 
        left = ceny - k + 1 - n
        
        
        psfs =  extract_psfs(data[i,...],centroids,window_size)
        
        #this needs to be changed ... 
        if redo_centroids:
            new_centroids = []
            for psf in psfs:
                new_centroids.append(centroid(psf)-(window_size/2,window_size/2))
            

        
       
        
        
                    
        #is this still usefull ?
        binary_regions = []
        thresh_arr = [] 
        watershed_regions_num_arr = []
        for psf in psfs:
            thresh = threshold_otsu(psf)
            binary = psf > thresh
            regions_num = label(binary).max()
            binary_regions.append(regions_num)
            thresh_arr.append(thresh)
            
            if regions_num ==1:
                watershed_regions_num = watershed_psf(binary)
                watershed_regions_num_arr.append(watershed_regions_num)
            else:
                watershed_regions_num_arr.append(None)
                
            
       
        
        M_dist_01 = distance_matrix(centroids,centroids)
        diag_idx = np.diag_indices(len(centroids))
        M_dist_01[diag_idx] = np.inf
        min_dist = M_dist_01.min(axis=0)
        
      
        
        
        df['min_dist'] = min_dist
        df['source'] = f'channel{i}'
        df['binary_regions'] = binary_regions
        df['threshold'] = thresh_arr
        df['waterhed_reginos_num'] = watershed_regions_num_arr
        df['avg_intensity'] = psfs.mean(axis=(1,2))
        df['bbox_anchor'] = list( zip(up,left))
        df['psfs'] = list(psfs)
        if redo_centroids:
            df['new_centroids'] = new_centroids
      
            
        dfs_tmp.append(df)

    all_channels_df = pd.concat(dfs_tmp, keys=['channel0','channel1','channel2','channel3'])
    
    return all_channels_df



def gaus_fit_from_df(df,
                     theta_guesses = [0,0,0,0],
                     sigma_x_guesses = [None,None,None,None,],
                     sigma_y_guesses = [None,None,None,None,],
                     apply_bounds = False):
    
        
    psfs = df['psfs'].values
    pixel_nr = len( psfs[0].flatten())
    channels = np.array( df['source'].values)
    
    orientation_arr = np.zeros_like(channels)
    sigma_x_arr = np.zeros_like(channels)
    sigma_y_arr = np.zeros_like(channels)
    
    #should definetly not be hardcoded

    orientation_arr[channels == 'channel0'] = theta_guesses[0]
    orientation_arr[channels == 'channel1'] = theta_guesses[1]
    orientation_arr[channels == 'channel2'] = theta_guesses[2]
    orientation_arr[channels == 'channel3'] = theta_guesses[3]
    
    sigma_x_arr[channels == 'channel0'] = sigma_x_guesses[0]
    sigma_x_arr[channels == 'channel1'] = sigma_x_guesses[1]
    sigma_x_arr[channels == 'channel2'] = sigma_x_guesses[2]
    sigma_x_arr[channels == 'channel3'] = sigma_x_guesses[3]
    
    sigma_y_arr[channels == 'channel0'] = sigma_y_guesses[0]
    sigma_y_arr[channels == 'channel1'] = sigma_y_guesses[1]
    sigma_y_arr[channels == 'channel2'] = sigma_y_guesses[2]
    sigma_y_arr[channels == 'channel3'] = sigma_y_guesses[3]
    
    bbox_anchor = np.array( df['bbox_anchor'].values)

    up,left = zip(* bbox_anchor)
    
    
    amplitude = []
    x0 = []
    y0 = []
    sigma_x = []
    sigma_y = []
    theta = []
    offset = []
    amplitude_cov = []
    x0_cov = []
    y0_cov = []
    sigma_x_cov = []
    sigma_y_cov = []
    theta_cov = []
    offset_cov = []
    det_cov = []
    photon_number = []
    residue = []
    cov_diago = []

     
    for psf,theta_guess,sigma_x_guess,sigma_y_guess in zip(psfs,orientation_arr,sigma_x_arr,sigma_y_arr):
        try:
            params,cov,fitted = fit_2d_gaussian(psf,theta_guess,sigma_x_guess,sigma_y_guess,apply_bounds = apply_bounds)
            
        except Exception as e:
            
            #print( e)
            
          
            amplitude.append(np.nan)
            x0.append(np.nan)
            y0.append(np.nan)
            sigma_x.append(np.nan)
            sigma_y.append(np.nan)
            theta.append(np.nan)
            offset.append(np.nan)
            
            amplitude_cov.append(np.nan)
            x0_cov.append(np.nan)
            y0_cov.append(np.nan)
            sigma_x_cov.append(np.nan)
            sigma_y_cov.append(np.nan)
            theta_cov.append(np.nan)
            offset_cov.append(np.nan)
            det_cov.append(np.nan)
            photon_number.append(np.nan)
            residue.append(np.nan)
            cov_diago.append(np.nan)
          
           
           
        else:
            
            amplitude.append(params[0])
            x0.append(params[1])
            y0.append(params[2])
            sigma_x.append(params[3])
            sigma_y.append(params[4])
            theta.append(params[5]%(2*np.pi))
            offset.append(params[6])
            
            photon_number.append(fitted.sum() - pixel_nr*params[6])
           
            amplitude_cov.append(np.sqrt(cov[0,0]))
            x0_cov.append(np.sqrt(cov[1,1]))
            y0_cov.append(np.sqrt(cov[2,2]))
            sigma_x_cov.append(np.sqrt(cov[3,3]))
            sigma_y_cov.append(np.sqrt(cov[4,4]))
            theta_cov.append(np.sqrt(cov[5,5]))
            offset_cov.append(np.sqrt(cov[6,6]))
            det_cov.append(np.linalg.det(cov))
            residue.append(np.sum(abs(fitted-psf)))
            cov_diago.append(cov[0,0]*cov[1,1]*cov[2,2]*cov[3,3]*cov[4,4]*cov[5,5])
            
    x0_gauss_global = np.array(left) + np.array(x0)
    y0_gauss_global = np.array(up) + np.array(y0)

    df['gauss_x0_global'] = x0_gauss_global
    df['gauss_y0_global'] = y0_gauss_global
    df['gauss_amp'] = amplitude
    df['gauss_x0'] = x0
    df['gauss_y0'] = y0
    df['gauss_sigma_x'] = sigma_x
    df['gauss_sigma_y'] = sigma_y
    df['gauss_theta'] = theta
    df['gauss_offset'] = offset
    df['total_int'] = photon_number
    df['residue'] = residue



    df['gauss_amp_cov'] = amplitude_cov
    df['gauss_x0_cov'] = x0_cov
    df['gauss_y0_cov'] = y0_cov
    df['gauss_sigma_x_cov'] = sigma_x_cov
    df['gauss_sigma_y_cov'] = sigma_y_cov
    df['gauss_theta_cov'] = theta_cov
    df['gauss_offset_cov'] = offset_cov
    df['gauss_det_cov'] = det_cov
    df['cov_diago'] = cov_diago

    return df
               



def find_psfs_and_clean(img,conv_kernels,ROI_masks,props,mask_nr,errosion_mask,area_threshold,plot_all,plot_final):
    '''
    

    Parameters
    ----------
    img : 2d array
        image from one channe;
    conv_kernels : list
        list of 2d-arrays that contains the convolutino kernels
    ROI_masks : 2d int array
        mask that labels all regions
    props : pd.DataFrame
        DataFrame that contains information about the channels
    mask_nr : int
        the channel that is currently investigated 
    errosion_size : int
        size of the errosion kernel, that is used to remove edge artefacts
    area_threshold : float
        minimal area value for a detected psfs to not be discarded
    plot_all : Bool
        turn on/off plotting the convlutions
    plot_final : TYPE
        turn on/off plotting the found regions

    Returns
    -------
    labeled : 2d array
        this array contains the labeled regions of the found psfs

    '''
    conv_filt = conv_psf_finder(img, conv_kernels,plot_all=plot_all,plot_final=plot_final)
    print('convolution done')
    erroded =  np.multiply(conv_filt,errosion_mask)
    opening = area_opening(erroded,area_threshold=area_threshold)
    labeled = label(opening)
    print('finished')
    
    return labeled

def get_affine_from_df(df,
                       channel_to_transform,
                       channel_fixed,
                       min_samples=3,
                       residual_threshold = 5,
                       max_trials = 1000,
                       return_points = False):
    
    closest_p_index =  df.loc[channel_to_transform,f'{channel_fixed}_closest_point_idx'].values.astype(dtype=int)

    channel_tt = df.loc[channel_to_transform]
    channel_fixed= df.loc[channel_fixed].iloc[closest_p_index]
    
    p_channel_tt = np.array(get_centroid_list(channel_tt))
    p_channel_fixed= np.array(get_centroid_list(channel_fixed))
  
    
    
    
    model_robust, inliers = ransac(
        (p_channel_tt,p_channel_fixed),
        AffineTransform,
        min_samples=min_samples,
        residual_threshold=residual_threshold,
        max_trials=max_trials
        )
    
    if return_points:

        p_channel_transformes = model_robust(p_channel_tt)
        
        return model_robust, inliers,p_channel_fixed,p_channel_tt,p_channel_transformes
    
    
    return model_robust, inliers



def show_affine_results(p1,p2,p2_transformed,inliers):
    
    outliers_idx = (inliers == False)
    
    
    p2_inliers = p2[inliers]
    p2_outliers = p2[outliers_idx]

    
    
    fig, ax  = plt.subplots()
    
    ax.scatter(p1[:,0],p1[:,1],color = 'r',marker = 'x',label='target position')
    ax.scatter(p2_inliers[:,0],p2_inliers[:,1],alpha = 0.5,color = 'b',marker = 'o',label = 'inital position (inliers)')
    ax.scatter(p2_transformed[:,0],p2_transformed[:,1],color = 'b',marker = '.',alpha = 0.5,label = 'transformed position')    
    ax.scatter(p2_outliers[:,0],p2_outliers[:,1],alpha = 0.5,color = 'yellow',marker = 'o',label = 'inital position (outliers)')
        
    ax.set_aspect('equal')
    
    plt.legend()
    

def show_movie_from_strack(z_stack):
    
    arr1 = z_stack[:,0,...]
    arr2 = z_stack[:,1,...]
    arr3 = z_stack[:,2,...]    
    arr4 = z_stack[:,3,...]
    # Create a 2x2 grid of subplots
    fig, axs = plt.subplots(2, 2,sharex=True,sharey=True,layout = 'constrained')
    ims = []
    
    # Unpack the subplots axes
    ax1, ax2, ax3, ax4 = axs.flatten()
    
    # Iterate over the length of the arrays (assumed to be the same for all)
    for i in range(len(arr1)):
        
        # Plot each frame in the respective subplot
        im1 = ax1.imshow(arr1[i, ...], animated=True)
        im2 = ax2.imshow(arr2[i, ...], animated=True)
        im3 = ax3.imshow(arr3[i, ...], animated=True)
        im4 = ax4.imshow(arr4[i, ...], animated=True)
        
        # Append the frames to the ims list
        ims.append([im1, im2, im3, im4])
    
    # Create the animation using the ims list
    ani = animation.ArtistAnimation(fig, ims, interval=250, blit=True, repeat_delay=0)
    
    return ani


def show_movie_from_strack_plus_scatter(z_stack,sactter_data):
    
    arr1 = z_stack[:,0,...]
    arr2 = z_stack[:,1,...]
    arr3 = z_stack[:,2,...]    
    arr4 = z_stack[:,3,...]
    # Create a 2x2 grid of subplots
    fig, axs = plt.subplots(2, 2,sharex=True,sharey=True,layout = 'constrained')
    ims = []
    
    # Unpack the subplots axes
    ax1, ax2, ax3, ax4 = axs.flatten()
    
    # Iterate over the length of the arrays (assumed to be the same for all)
    for i in range(len(arr1)):
        
        # Plot each frame in the respective subplot
        im1 = ax1.imshow(arr1[i, ...], animated=True)
        im2 = ax2.imshow(arr2[i, ...], animated=True)
        im3 = ax3.imshow(arr3[i, ...], animated=True)
        im4 = ax4.imshow(arr4[i, ...], animated=True)
        
        # Append the frames to the ims list
        ims.append([im1, im2, im3, im4])
    
    # Create the animation using the ims list
    ani = animation.ArtistAnimation(fig, ims, interval=250, blit=True, repeat_delay=0)
    
    return ani

def link_centroid_lists(ls1,ls2,max_dist=5):
    
    '''orders ls2 so the points have a minimal distance to ls1,
    points from ls1 that dont have a corresponding point within max_dist are dropped'''
    
    dist_M = distance_matrix(ls1, ls2)
    ls2_new = np.zeros((len(ls1),2))
    ls2_new[...] = np.nan
    min_dist = dist_M.min(axis = 1)
    bool_array = min_dist < max_dist
    dist_M_prime = dist_M[bool_array,...]
    ls2_ind = dist_M_prime.argmin(axis= 1)
    
    ls2_new[bool_array] = np.array(ls2)[ls2_ind,:]
    return ls2_new


def xy_to_polar_angle(x,y):
    
    x=np.array(x)
    y=np.array(y)
    
    angles = np.array( np.arctan(y/x))
    
    first_quad = (x > 0) *  (y > 0)
    second_quad = (x < 0) *  (y > 0)
    third_quad = (x < 0) *  (y < 0)
    fourth_quad = (x > 0) *  (y < 0)
    
    angles[second_quad] = np.pi + angles[second_quad]
    angles[third_quad] = np.pi + angles[third_quad]
    angles[fourth_quad] = 2*np.pi + angles[fourth_quad]

    return angles

