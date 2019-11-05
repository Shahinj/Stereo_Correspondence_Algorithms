import numpy as np
from numpy.linalg import inv
from scipy.ndimage.filters import *

def stereo_disparity_fast(Il, Ir, bbox, maxd):
    """
    Fast stereo correspondence algorithm.

    This function computes a stereo disparity image from left stereo 
    image Il and right stereo image Ir. Only disparity values within
    the bounding box region are evaluated.

    Parameters:
    -----------
    Il    - Left stereo image, m x n pixel np.array, greyscale.
    Ir    - Right stereo image, m x n pixel np.array, greyscale.
    bbox  - 2x2 np.array, bounding box, relative to left image, from top left
            corner to bottom right corner (inclusive).
    maxd  - Integer, maximum disparity value; disparities must be within zero
            to maxd inclusive (i.e., don't search beyond rng)

    Returns:
    --------
    Id  - Disparity image (map) as np.array, same size as Il, greyscale.
    """
    # Hints:
    #
    #  - Loop over each image row, computing the local similarity measure, then
    #    aggregate. At the border, you may replicate edge pixels, or just avoid
    #    using values outside of the image.
    #
    #  - You may hard-code any parameters you require in this function.
    #
    #  - Use whatever window size you think might be suitable.
    #
    #  - Don't optimize for runtime (too much), optimize for clarity.

    #--- FILL ME IN ---
    
    #get the coordinates of the bounding box
    x_top_left , y_top_left  =  bbox[:,0]
    x_bot_right, y_bot_right =  bbox[:,1]
        
    #get the overlapping section
    matching_patch = Il[ y_top_left:y_bot_right, x_top_left:x_bot_right]
    m,n = matching_patch.shape
    
    #initialize the results
    Id = np.zeros(Il.shape)
    
    #suppression and patch window, to combine pixel values. This helps the algorithm run faster
    window = 5
    #loop through all the overlapping section pixels
    for xl in range(x_top_left,x_bot_right - window, window):
        for yl in range(y_top_left,y_bot_right - window, window):
            #get the patch in the left image
            to_match = Il[yl:yl+window, xl:xl+window]
            #initialize the maximum difference to be infinity
            lowest_diff = np.inf
            #initialize the disparity
            disparity   = 0
            #loop through possible values in the right image, and find the best patch. (just need to do x because fronto-parallel picture)
            for xr in range(xl, xl - maxd - 1, -1):
                #to avoid errors
                if(xr - window < 0):
                    continue
                #find the best match for that window in the right picture
                on_the_right = Ir[yl:yl+window, xr-window:xr]
                #compute the sad score
                sad = np.sum(np.abs(to_match.flatten() - on_the_right.flatten()))
                #update the disparity and the best score, if the newly computed score is better
                if(sad < lowest_diff):
                    lowest_diff = sad
                    disparity = xl - xr
            #assign the disparity to the window
            Id[yl:yl+window,xl:xl+window] = disparity
    #------------------

    return Id