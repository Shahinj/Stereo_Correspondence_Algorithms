import numpy as np
from numpy.linalg import inv
from scipy.ndimage.filters import *

def stereo_disparity_best(Il, Ir, bbox, maxd):
    """
    Best stereo correspondence algorithm.

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
    
    #define parameters
    #supression is to combine the pixel values in the vicinity to be the same
    supression_window = 3
    #threshold for acceptance of difference between pixel values
    thresh = 5
    #loop through all the overlapping section pixels
    for yl in range(y_top_left,y_bot_right, supression_window):
        #initialize the caching data structure, this is to save up calculations/slicing that we do, so we don't do them multiple times
        all_windows = {}
        for xl in range(x_top_left,x_bot_right, supression_window):
            #find the best match for that window in the right picture. The idea is that: first find all the pixels that match, then increase the window by 1, and then within those matching pixels, compare the windows around them, if they match again, make the window larger, until one pixel wins. 
            window = 0
            #get the initial set of x's in the right picture that can be the correct match
            possible_matches = []
            possible_xrs = [xr for xr in range(np.clip(xl,0,xl), xl - maxd - 1, -1)]
            #do the algorithm above, until one wins, or the window becomes to large. If after the window becoming 2 large there is still multiple xs that match, then there is practically no difference in which one we take
            while(len(possible_xrs) != 1  and window <= 7):
                #get the left image patch
                to_match     = Il[yl-window: yl+window+1, xl-window: xl+window+1].flatten()                
                #initialize the best score to be 0. Note that with the new scoring algorithm, we are interested in a patch that has the highest number of pixels being the range with their corresponding pixel.
                best_score = 0
                #loop through possible x's and calculate the score
                for xr in possible_xrs:
                    #check if we already computed/sliced the window around that xr, with that window size
                    if((xr, window) not in all_windows):
                        on_the_right = Ir[yl-window: yl+window+1, xr-window: xr+window+1].flatten()
                        all_windows[(xr, window)] = on_the_right
                    else:
                        #if yes, use that
                        on_the_right = all_windows[(xr, window)]
                    #This is to avoid errors when we go over the bounds
                    try:
                        score = (abs(to_match-on_the_right) < thresh).sum()
                    except:
                        break
                    
                    #if we found a better patch than the rest, the rest become useless, delete them and just add this new one to the contenders list. Also update the best score so far
                    if(score > best_score):
                        possible_matches = [xr]
                        best_score = score
                    #if we get the same score, add it to the contenders list. Check if we already added it to the list to avoid multiple addition
                    elif(score == best_score and xr not in possible_matches):
                        possible_matches.append(xr)
                
                #update the list of contenders
                possible_xrs = possible_matches
                #increase the window size for the new tests
                window +=1
                
            #once we are done, we have either the best match, or a bunch matches that are equally good. Set the disparity to be the first element then.
            disparity = xl - possible_xrs[0]
            #do the supression and merging
            Id[yl:yl+supression_window,xl:xl+supression_window] = disparity
    
    #reduce the noise, and make it smoother
    Id =  median_filter(Id,10)

    #------------------

    return Id
