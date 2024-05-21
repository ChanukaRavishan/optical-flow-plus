import cv2
import numpy as np

sequence_folder = 'KITTI Dataset/training/image_2'
ground_truth_folder = 'KITTI Dataset/training/flow_occ'
output_folder = 'path/to/save/output'

image1 = cv2.imread(sequence_folder + '/000010_10.png')
image2 = cv2.imread(sequence_folder + '/000010_11.png')
ground_truth_flow = cv2.imread(ground_truth_folder + '/000010_10.png')

ground_truth_flow = ground_truth_flow.astype(np.float32) / 255.0

gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, pyr_scale=0.5, levels=5, winsize=11, iterations=5,
                                     poly_n=5, poly_sigma=1.1, flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN)

# Scale the flow values back to the range [0, 255]
scaled_flow = flow * 255.0

# Convert to uint8
scaled_flow = scaled_flow.astype(np.uint8)

# Save the computed flow as an image
output_path = '/000010_10_computed_flow.png'  # Adjust the filename as needed
cv2.imwrite(output_path, scaled_flow)

print("Computed flow saved at:", output_path)
