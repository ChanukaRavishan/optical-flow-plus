
import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import convolve
import evaluating_codes.average_end_point_error as average_end_point_error


def get_first_order_derivatives(img1, img2):
    x_kernel = np.array([[-1, 1], [-1, 1]]) * 0.25
    y_kernel = np.array([[-1, -1], [1, 1]]) * 0.25
    t_kernel = np.ones((2, 2)) * 0.25

    fx = convolve(img1,x_kernel) + convolve(img2,x_kernel)
    fy = convolve(img1, y_kernel) + convolve(img2, y_kernel)
    ft = convolve(img1, -t_kernel) + convolve(img2, t_kernel)

    return [fx,fy, ft]

def get_second_Order_derivatives(fx, fy, ft):
    
    dxx_kernel = np.array([[1, -2, 1]])
    dyy_kernel = np.array([[1], [-2], [1]])
    dtt_kernel = np.array([[1, -2, 1]])


    fxx = convolve(fx, dxx_kernel)
    fyy = convolve(fy, dyy_kernel)
    ftt = convolve(ft, dtt_kernel)

    return [fxx,fyy, ftt]

def get_cross_derivatives(fx, fy, ft):
    
    dxy_kernel = np.array([[1, 1], [-1, -1]]) * 0.25
    dyt_kernel = np.array([[1], [1], [-1], [-1]]) * 0.25
    dtx_kernel = np.array([[1, -1], [1, -1]]) * 0.25

    # Calculate the cross derivatives
    dfxdy = convolve(fx, dxy_kernel)
    dfydt = convolve(fy, dyt_kernel)
    dftdx = convolve(ft, dtx_kernel)

    return [dfxdy, dfydt, dftdx]

def optical_flow_estimation(name1, name2, delta):
    beforeImg = cv2.cvtColor(name1, cv2.COLOR_BGR2GRAY)
    afterImg  = cv2.cvtColor(name2, cv2.COLOR_BGR2GRAY)

    beforeImg = cv2.cvtColor(name1, cv2.COLOR_BGR2GRAY).astype(float)
    afterImg = cv2.cvtColor(name2, cv2.COLOR_BGR2GRAY).astype(float)

    beforeImg  = cv2.GaussianBlur(beforeImg, (5, 5), 0)
    afterImg = cv2.GaussianBlur(afterImg, (5, 5), 0)
    
    u = np.zeros((beforeImg.shape[0], beforeImg.shape[1]))
    v = np.zeros((beforeImg.shape[0], beforeImg.shape[1]))
    fx, fy, ft = get_first_order_derivatives(beforeImg, afterImg)
    fxx,fyy,ftt = get_second_Order_derivatives(fx, fy, ft)
    dfxdy, dfydt, dftdx = get_cross_derivatives(fx, fy, ft)
    
    
    avg_kernel = np.array([[1 / 12, 1 / 6, 1 / 12],
                            [1 / 6, 0, 1 / 6],
                            [1 / 12, 1 / 6, 1 / 12]], float)
    iter_counter = 0
    
    while True:
        iter_counter += 1
        u_avg = convolve(u, avg_kernel)
        v_avg = convolve(v, avg_kernel)

        p = (fx + 0.5*fxx + dfxdy) * u_avg + (fy + 0.5*fyy + dfydt) * v_avg + (ft + 0.5*ftt + dftdx)
        d = 1000 + fx**2 + fy**2
        prev = u

        u = u_avg - fx * (p / d)
        v = v_avg - fy * (p / d)

        diff = np.linalg.norm(u - prev, 2)
        #converges check (at most 300 iterations)
        if  diff < delta or iter_counter > 300:
            # print("iteration number: ", iter_counter)
            break

    return [u, v]


sequence_folder = 'KITTI Dataset/training/image_2'
ground_truth_folder = 'KITTI Dataset/training/flow_occ'
image1 = cv2.imread(sequence_folder + '/000111_10.png')
image2 = cv2.imread(sequence_folder + '/000111_11.png')
ground_truth_flow = cv2.imread(ground_truth_folder + '/000111_10.png')
ground_truth_flow_image_path = ground_truth_folder + '/000111_10.png'

#ground_truth_flow = (ground_truth_flow.astype(np.float32) - (2^15) ) / 64.0

u,v = optical_flow_estimation(image1, image2, delta = 10**-1)

#endpoint_error = np.sqrt((u - ground_truth_flow[:, :, 0]) ** 2 + (v - ground_truth_flow[:, :, 1]) ** 2)

#average_endpoint_error = np.mean(endpoint_error)

#print("Average Endpoint Error:", average_endpoint_error)

aee = average_end_point_error.compute_aee(ground_truth_flow_image_path, u, v)

print(f"Average Endpoint Error from ++: {aee}")


