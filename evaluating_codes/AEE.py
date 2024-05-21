import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the KITTI dataset sequence and corresponding optical flow ground truth
sequence_folder = 'KITTI Dataset/testing/image_2'
ground_truth_folder = 'KITTI Dataset/training/disp_occ_0'
image1 = cv2.imread(sequence_folder + '/000001_10.png')
image2 = cv2.imread(sequence_folder + '/000001_11.png')
ground_truth_flow = cv2.imread(ground_truth_folder + '/000001_10.png')

# Convert the ground truth flow to floating-point format
ground_truth_flow = ground_truth_flow.astype(np.float32) / 255.0

gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, pyr_scale=0.5, levels=5, winsize=11, iterations=5,
                                    poly_n=5, poly_sigma=1.1, flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN)

# Separate the flow into u and v components
u = flow[:, :, 0]
v = flow[:, :, 1]

# Evaluate the endpoint error between the computed flow and ground truth flow
endpoint_error = np.sqrt((u - ground_truth_flow[:, :, 0]) ** 2 + (v - ground_truth_flow[:, :, 1]) ** 2)

