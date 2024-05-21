import cv2
import numpy as np
from PIL import Image


def calAee(ground_truth, img1, img2):

    # To convert the u-/v-flow into floating point values, convert the value to float, subtract 2^15 and divide the result by 64.0:

    #flow_u(u,v) = ((float)I(u,v,1)-2^15)/64.0;
    #flow_v(u,v) = ((float)I(u,v,2)-2^15)/64.0;
    #valid(u,v)  = (bool)I(u,v,3);

    flow_u = ground_truth[:, :, 0]  # u-component (first channel)
    flow_v = ground_truth[:, :, 1]  # v-component (second channel)
    validity = ground_truth[:, :, 2].astype(bool)  # validity (third channel, converted to boolean)

    # Convert u and v components to floating-point values
    flow_u = (flow_u.astype(float) - 2**15) / 64.0
    flow_v = (flow_v.astype(float) - 2**15) / 64.0


    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, pyr_scale=0.5, levels=5, winsize=11, iterations=5,
                                         poly_n=5, poly_sigma=1.1, flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN)

    u = flow[:, :, 0]
    v = flow[:, :, 1]

    #if validity.any():
    #    valid_pixels = validity

    valid_pixels = validity

    error_u = u[valid_pixels] - flow_u[valid_pixels]
    error_v = v[valid_pixels] - flow_v[valid_pixels]

    # Calculate average endpoint error
    aee = np.mean(np.sqrt(error_u**2 + error_v**2))
    print("Average Endpoint Error (AEE) (manual):", aee)
    

def genpng(image1, image2):

    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, pyr_scale=0.5, levels=5, winsize=11, iterations=5,
                                         poly_n=5, poly_sigma=1.1, flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN)

    u = flow[:, :, 0]
    v = flow[:, :, 1]



    validity = (u != 0) & (v != 0)

    flow_u = (u * 64.0) + 2**15
    flow_v = (v * 64.0) + 2**15

    ground_truth = np.stack([flow_u, flow_v, validity], axis=2)

    #ground_truth = ground_truth.astype(np.uint16)
    
    output_path = '000010_10_computed_flow.png'  # Adjust the filename as needed
    cv2.imwrite(output_path, ground_truth)


def comparingpng(png, ground_truth):

    # ground truth

    flow_u = ground_truth[:, :, 0]
    flow_v = ground_truth[:, :, 1] 
    validity = ground_truth[:, :, 2].astype(bool)

    flow_u = (flow_u.astype(float) - 2**15) / 64.0
    flow_v = (flow_v.astype(float) - 2**15) / 64.0

    #created png

    u = png[:, :, 0]
    v = png[:, :, 1] 
    val = png[:, :, 2].astype(bool)

    u = (u.astype(float) - 2**15) / 64.0
    v = (v.astype(float) - 2**15) / 64.0


    valid_pixels = validity

    error_u = u[valid_pixels] - flow_u[valid_pixels]
    error_v = v[valid_pixels] - flow_v[valid_pixels]

    # Calculate average endpoint error
    aee = np.mean(np.sqrt(error_u**2 + error_v**2))
    print("Average Endpoint Error (AEE) gen vs ground:", aee)




sequence_folder = 'KITTI Dataset/training/image_3'
ground_truth_folder = 'KITTI Dataset/training/flow_occ'
image1 = cv2.imread(sequence_folder + '/000006_10.png')
image2 = cv2.imread(sequence_folder + '/000006_11.png')
ground_truth_flow = cv2.imread(ground_truth_folder + '/000006_10.png')


calAee(ground_truth_flow, image1, image2)

#genpng(image1, image2)

png = cv2.imread('Results/xxxx.png')

comparingpng(png, ground_truth_flow) 


