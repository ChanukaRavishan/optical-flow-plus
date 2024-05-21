import cv2
import numpy as np


def convert_flow_to_channels(flow_image):

    flow = cv2.imread(flow_image, cv2.IMREAD_UNCHANGED)

    
    u_channel = flow[:, :, 0]
    v_channel = flow[:, :, 1]
    valid_channel = flow[:, :, 2]

    
    flow_u = ((u_channel.astype(float) - 2**15) / 64.0)
    flow_v = ((v_channel.astype(float) - 2**15) / 64.0)

    
    optical_flow_map = np.zeros_like(flow, dtype=np.float32)

    # Set u and v channels
    optical_flow_map[:, :, 0] = flow_u
    optical_flow_map[:, :, 1] = flow_v

    # Set valid channel
    optical_flow_map[:, :, 2] = valid_channel.astype(float)

    return optical_flow_map



def compute_aee(ground_truth_flow, computed_flow_u, computed_flow_v):

    ground_truth_flow_map = convert_flow_to_channels(ground_truth_flow)

    # Extract u, v, and valid channels from the ground truth
    gt_u_channel = ground_truth_flow_map[:, :, 0]
    gt_v_channel = ground_truth_flow_map[:, :, 1]
    valid_channel = ground_truth_flow_map[:, :, 2]

    # Compute endpoint error only for valid pixels
    valid_pixels = valid_channel > 0

    # Compute the squared endpoint error
    epe_squared = (gt_u_channel - computed_flow_u) ** 2 + (gt_v_channel - computed_flow_v) ** 2

    # Sum up the squared errors for valid pixels
    valid_epe_squared_sum = np.sum(epe_squared[valid_pixels])

    # Count the number of valid pixels
    num_valid_pixels = np.sum(valid_pixels)

    # Compute the average endpoint error
    aee = np.sqrt(valid_epe_squared_sum / num_valid_pixels)

    return aee


sequence_folder = 'KITTI Dataset/training/image_2'
ground_truth_folder = 'KITTI Dataset/training/flow_occ'
ground_truth_flow_image_path = ground_truth_folder + '/000100_10.png'

image1 = cv2.imread(sequence_folder + '/000111_10.png')
image2 = cv2.imread(sequence_folder + '/000111_11.png')

gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

computed_flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, pyr_scale=0.5, levels=5, winsize=11, iterations=5,
                                              poly_n=5, poly_sigma=1.1, flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN)

computed_flow_u = computed_flow[:, :, 0]
computed_flow_v = computed_flow[:, :, 1]

aee = compute_aee(ground_truth_flow_image_path, computed_flow_u, computed_flow_v)

print(f"Average Endpoint Error: {aee}")
