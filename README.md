

***Vehicle Detection Project***

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[1_car_not_car]: ./write_up_img/1_car_not_car.png
[2_Car_hog]: ./write_up_img/2_Car_hog.png
[2_not_car_hog]: ./write_up_img/2_not_car_hog.png
[3_sliding_window]: ./write_up_img/3_sliding_window.png
[5_bboxes_and_heat]: ./write_up_img/5_bboxes_and_heat.png
[6_label]: ./write_up_img/6_label.png
[7_label_heat_findcar]: ./write_up_img/7_label_heat_findcar.png
[video1]: ./write_up_img.mp4

---

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the function `extract_features()` in `proj5_functions.py`.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:
![alt text][1_car_not_car]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][2_Car_hog]
![alt text][2_not_car_hog]


#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and find `color_space` and `hog_channel` has significant impact training results. I tried `HSV,LUV,YUV,YCrCb` color spaces. It proves `YCrCb` is the best fit for this task. `hog_channel` needs to utilize all channels to take advantage full potential power of HOG algorithm. Parameters setting is at line 58~70 in `pipeline2.py`.


#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a SVM by using function `GridSearchCV()` for carrying out automatic paramter search. The parameters I got for SVM is `(Kernal:'rbf', C:10)`. The code is at lines 107~121 in `pipeline2.py`

### Sliding Window Search

#### Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

Since image part of road is mainly at bottom half of the whole picture, my search window only slides through bottom half part every frames. The scale I used is `[1,1.5]`. Tried smaller scale like `0.8` and larger scale like `2.0`. No obvious improvement observed and more scales could cause false dectection and slow down video processing. Thuse, only `1` and `1.5` are used in my code.

![alt text][3_sliding_window]


#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][7_label_heat_findcar]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap. The result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

![alt text][7_label_heat_findcar]




