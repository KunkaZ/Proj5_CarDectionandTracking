from glob import glob as glob2
from proj5_functions import *
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.grid_search import GridSearchCV
from skimage.feature import hog
import pickle
from scipy.ndimage.measurements import label

from sklearn.model_selection import train_test_split
# from sklearn.cross_validation import train_test_split
DEBUG_SWITCH = 1
##---------------------------------Step 0: setup data set---------------------------------------------------
# Get training data images paths
vehicles_data_path      = '../vehicles/'
non_vehicles_data_path  = '../non-vehicles/'
output_images_path      = '../output_images/'

paths_vehicles      = glob2(vehicles_data_path+'*/')
paths_non_vehicles  = glob2(non_vehicles_data_path+'*/')

cars = []
notcars = []
for path in paths_vehicles:
    temp_img = glob.glob(path+'*.*')
    cars.extend(temp_img)

for path in paths_non_vehicles:
    temp_img = glob.glob(path+'*.*')
    notcars.extend(temp_img)
test_cars = cars
test_notcars = notcars
# test_cars = cars[0:3000]
# test_notcars = notcars[0:3000]
print('--------------------1. setup data set--------------------------------')
print('Total num of car images:',len(cars))
print('Total num of notcar images:',len(notcars))


##---------------------------------Step 2: extract features---------------------------------------------------
#
# 1) spatial
# 2) HOG
# 3) 
# exit()

print('-------------------------2. SVM model training---------------------------')
# saved_X_scaler = load_svm_model('X_scaler1000111.sav')

saved_model_file = 'trained_modelxxx.p'
saved_model = load_svm_model(saved_model_file)



if saved_model == False :
    #TODO add HSV
    # color_space = 'HSV' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    color_space = 'HSV' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    orient = 9  # HOG orientationsd
    pix_per_cell = 8 # HOG pixels per cell
    cell_per_block = 2 # HOG cells per block
    hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
    spatial_size = (32, 32) # Spatial binning dimensions
    hist_bins = 64    # Number of histogram bins
    spatial_feat = True # Spatial features on or off
    hist_feat = True # Histogram features on or off
    hog_feat = True # HOG features on or off
    y_start_stop = [360,None] # Min and max in y to search in slide_window()
    print('Start SVM model training session:')

    print('Extract car features...')
    car_features = extract_features(test_cars, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)

    print('Extract notcar features...')
    notcar_features = extract_features(test_notcars, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)

    X = np.vstack((car_features, notcar_features)).astype(np.float64)                        
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)
    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))
    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.2, random_state=rand_state, shuffle=True)

    print('Using:',orient,'orientations',pix_per_cell,
        'pixels per cell and', cell_per_block,'cells per block')
    print('Feature vector length:', len(X_train[0]))
    print('Training SVC...')
    # Use a linear SVC 

    parameters = {'kernel':('linear', 'rbf'), 'C':[1,3,5,7,9, 10]}
    svr = SVC()
    t=time.time()
    clf = GridSearchCV(svr, parameters)  
    clf.fit(X_train, y_train)
    print('best params:',clf.best_params_)
    # Check the training time for the SVC


    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to train SVC...')
    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(clf.score(X_test, y_test), 4))
    # Check the prediction time for a single sample
    t=time.time()


    filename = 'trained_model.p'
    svc_para = {'clf':clf,
                'X_scaler':X_scaler,
                'color_space': color_space, # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
                'orient': orient,  # HOG orientations
                'pix_per_cell': pix_per_cell, # HOG pixels per cell
                'cell_per_block': cell_per_block, # HOG cells per block
                'hog_channel': hog_channel, # Can be 0, 1, 2, or "ALL"
                'spatial_size': spatial_size, # Spatial binning dimensions
                'hist_bins': hist_bins,    # Number of histogram bins
                'spatial_feat': spatial_feat, # Spatial features on or off
                'hist_feat': hist_feat, # Histogram features on or off
                'hog_feat': hog_feat, # HOG features on or off
                'y_start_stop': y_start_stop} # Min and max in y to search in slide_window()
    pickle.dump(svc_para, open(filename, 'wb'))


    # print('Load saved SVM model')
    # loaded_model = pickle.load(open(filename, 'rb'))
    # result = loaded_model.score(X_test, y_test)
    # print('Test Accuracy of saved SVC = ',result)
else:
    print('Saved model loaded.')
    clf             = saved_model["clf"]
    X_scaler        = saved_model["X_scaler"]
    color_space     = saved_model["color_space"]
    orient          = saved_model["orient"]
    pix_per_cell    = saved_model["pix_per_cell"]
    cell_per_block  = saved_model["cell_per_block"]
    hog_channel     = saved_model["hog_channel"]
    spatial_size    = saved_model["spatial_size"]
    hist_bins       = saved_model["hist_bins"]
    spatial_feat    = saved_model["spatial_feat"]
    hog_feat        = saved_model["hog_feat"]
    hist_feat       = saved_model["hist_feat"]
    y_start_stop    = saved_model["y_start_stop"]
    
# exit()


print('-------------------------3. test on single image---------------------------')

image = mpimg.imread('../test_images/test5.jpg')
process_image(image)
"""
use_HOG_subsampling = 1
if not use_HOG_subsampling:
    draw_image = np.copy(image)
    # Uncomment the following line if you extracted training
    # data from .png images (scaled 0 to 1 by mpimg) and the
    # image you are searching is a .jpg (scaled 0 to 255)
    image = image.astype(np.float32)/255
    hot_windows = []
    xy_windows = [[128,128],[96,96],[64,64]]
    for xy_window in xy_windows: 
        windows = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop, 
                            xy_window=xy_window, xy_overlap=(0.5, 0.5))
        # print(windows)

        hot_window = search_windows(image, windows, clf, X_scaler, color_space=color_space, 
                                spatial_size=spatial_size, hist_bins=hist_bins, 
                                orient=orient, pix_per_cell=pix_per_cell, 
                                cell_per_block=cell_per_block, 
                                hog_channel=hog_channel, spatial_feat=spatial_feat, 
                                hist_feat=hist_feat, hog_feat=hog_feat) 
        hot_windows.extend(hot_window)                                
    # print(hot_windows)
    out_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)           

    plt.imshow(out_img)
    plt.show()

# test 2 use hog sub-sampling
else:
    #TODO: return hot_windos from find cars;
    #TODO: modifiy color_space so I can pass it into find_cars
    print('use hog sub-sampling')
    ystart  = 400
    ystop   = 656
    scales  = {0.5,1,1.5,2}
    print('search with scale',scales)
    out_img, hot_windows = find_cars(image, ystart, ystop, scales, clf, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)

    plt.imshow(out_img)

    plt.show()
# draw heat map

heat = np.zeros_like(out_img[:,:,0]).astype(np.float)
# print(hot_window)
heat = add_heat(heat,hot_windows)

# Apply threshold to help remove false positives
heat = apply_threshold(heat,1)

# Visualize the heatmap when displaying    
heatmap = np.clip(heat, 0, 255)

labels = label(heatmap)
draw_img = draw_labeled_bboxes(np.copy(image), labels)

fig = plt.figure()
plt.subplot(121)
plt.imshow(draw_img)
plt.title('Car Positions')
plt.subplot(122)
plt.imshow(heatmap, cmap='hot')
plt.title('Heat Map')
fig.tight_layout()
plt.show()
exit()
"""
######------------------
#Step 2: Combine features


# Feature normazalization

#Step 2: train classifier with test images

# split data set into training and validation


#Step 3: Process video

# Outside loop sliding window
# 1)sliding window 
# 2)use heat map to filter  overlapping and false positive
# 3)(opt)estimate car move from frames to frappes
