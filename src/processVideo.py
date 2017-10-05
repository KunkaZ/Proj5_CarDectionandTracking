from proj5_functions import *
from moviepy.editor import VideoFileClip





# Function for using previous multiple frames to generate heat map for current frame
def MultiFrameDetection(new_detections, old_detection):

	# If no cars add any detections as a new vehicle and put in old_detection
	if len(old_detection) == 0:
		for new_detection in new_detections:
			old_detection.append(vehicle(new_detection[0],new_detection[1],new_detection[2],new_detection[3]))

	# If no detections but old_detection is not empty then demote all vehicle detections
	elif len(new_detections) == 0:
		for car in old_detection:
			if car.NegUpdate():
				old_detection.remove(car)

	# Both old_detection and new detections are non-empty and need to do comparisons to pair results
	else:
		# Keep track of what detections need to be made into a new vehicle
		new_detection_update = np.zeros((len(new_detections)))
		# Keep track of what cars need to get demoted because they didnt have a new track
		remove_update = np.zeros((len(old_detection)))
        # use a matrix to track new detections and existing detections
		# new detections are stored on the rows and exisitng detections on the columbs
		# The values we are storing between a new/old caputre pair is the rank defined as distance between positions
		grid = np.zeros((len(new_detections),len(old_detection)+1))

		# Iterate through new detections
		for n in range(len(new_detections)):
			# Iterate through old detections
			for o in range(len(old_detection)):
				# Measure the distance between new detection and all old detections
				grid[n,o] = old_detection[o].Rank(new_detections[n][0],new_detections[n][1])
			# Store the old detection that was closest to the new detection 
			grid[n,len(old_detection)] = np.argmin(grid[n][:-1])
		# Iterate through old detections
		for o in range(len(old_detection)):
			# Iterate through new detections
			for n in range(len(new_detections)):
				# Mark all ranks as very high if new/old detection pair didnt match
				if grid[n][-1] != o:
					grid[n,o] = 1000

			# If the rank is good between new/old detection pair
			if np.amin(grid[:,o]) < 1000:
				# Establish a link to new/old detection and update new detected positon and window dimensions
				get_new_detection = new_detections[np.argmin(grid[:,o])]
				old_detection[o].PosUpdate(get_new_detection[0],get_new_detection[1],get_new_detection[2],get_new_detection[3])
				new_detection_update[np.argmin(grid[:,o])] = 1
			else:
				# Else old detection did not have a new detection
				remove_update[o] = 1
		# Check what new detections did not have a matched old detection and create a new vehicle out of it, and add it in old_detection
		for i in range(len(new_detection_update)):
			if new_detection_update[i] == 0:
				get_new_detection = new_detections[i]
				old_detection.append(vehicle(get_new_detection[0],get_new_detection[1],get_new_detection[2],get_new_detection[3]))

		# Look through old detections that did not have a new detection and demote them, if the negtative detections in a row is
		# greater than some threshold we delete the old detection from the old_detection
		remove_index = []
		for car in range(len(remove_update)):
			if remove_update[car] == 1 and old_detection[car].NegUpdate():
				remove_index.append(car)

		old_detection = np.delete(old_detection,remove_index)
		old_detection = old_detection.tolist()

	return old_detection

# Process each frame of the video
def process_image_rgb(image):
	# We want to modify old_detections and heatmaps from this function
	global old_detections, heatmaps
	# Create a general heatmap
	heatmap = np.zeros_like(image[:,:,0])
	# Iterate through different scale values
	for scale in np.arange(1,2,0.5):
		# Create heat maps for different scales in function to both search and classify
		out_img, heat_map = find_cars(image, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
		# Add heat to heatmap from different scales
		heatmap += heat_map
	
	# Add general heatmap to list of heatmaps
	heatmaps.append(heatmap)
    # Add previous 5 heatmaps into 1 heatmap
	heatmap = MultiFrameHeatMap(image,4)
    addheat = np.zeros_like(heatmap[:,:,0])
    nframe = min(5,len(addheat))
    for i in np.arange(1,nframe):
        addheat += addheat[-i]
        heatmap = addheat

	# Apply threshold to remove false positives from heatmap
	heatmap = apply_threshold(heatmap,4)
	# Get the box positions from heatmap
	labels = label(heatmap)

	# Store box parameters for each box as a new detection
	new_detections = []
	for car_number in range(1, labels[1]+1):
		nonzero = (labels[0] == car_number).nonzero()
		nonzeroy = np.array(nonzero[0])
		nonzerox = np.array(nonzero[1])

		xpos = np.min(nonzerox)
		ypos = np.min(nonzeroy)

		width = np.max(nonzerox)-xpos
		height = np.max(nonzeroy)-ypos

		new_detections.append([xpos,ypos,width,height])

	# Combine old detections with new detections
	old_detections = MultiFrameDetection(new_detections,old_detections)

	# Draw boxes on a copy of the image
	draw_img = np.copy(image)
    for car_number in range(len(old_detection)):
        if old_detection[car_number].detected:
            bbox = old_detection[car_number].Box()
            cv2.rectangle(draw_img, bbox[0], bbox[1], (0, 0, 255), 6) 


	return draw_img


# Load up saved data from SVM post training
dist_pickle = pickle.load( open("train_model.p", "rb" ) )
clf = dist_pickle["clf"]
X_scaler = dist_pickle["scaler"]
orient = dist_pickle["orient"]
pix_per_cell = dist_pickle["pix_per_cell"]
cell_per_block = dist_pickle["cell_per_block"]
spatial_size = dist_pickle["spatial_size"]
hist_bins = dist_pickle["hist_bins"]

# Define the y range to search for cars
ystart = 400
ystop = 656
# Save off heatmaps and car detected
heatmaps = []
old_detections = []


Input_video = '..\project_video.mp4'
video_output = '..\project_video_output.mp4'
clip1 = VideoFileClip(Input_video)
video_clip = clip1.fl_image(process_image_rgb) #NOTE: this function expects color images!!
video_clip.write_videofile(video_output, audio=False)