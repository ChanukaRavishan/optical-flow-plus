import cv2
import numpy as np
from matplotlib import pyplot as plt

#to apply gausian laplace to n dimensional array
from scipy.ndimage import convolve
import os
import time
#from argparse import ArgumentParser


#computing magnitude in each 8 pixels. return magnitude average
#input: u,v vectors
#output: magnitude average

start_time = time.time()
scale = 3

def get_magnitude(u, v):
    dy = v[::8,::8] * scale
    dx = u[::8,::8] * scale
    magnitude = np.sqrt(dx**2 + dy**2)
    mag_avg = np.mean(magnitude)
    return mag_avg


#Quiver Plot
#input: u,v vectors, before image
#output: quiver plot
#draws quiver plot on the before image

def quiver(u, v, beforeImg, save_path='resulting_image.png'):
    fig, ax = plt.subplots()
    ax.imshow(beforeImg, cmap='gray')

    magnitudeAvg = get_magnitude(u, v)

    for i in range(0, u.shape[0], 8):
        for j in range(0, u.shape[1], 8):
            dy = v[i, j] * scale
            dx = u[i, j] * scale
            magnitude = (dx**2 + dy**2)**0.5
            # draw only significant changes
            if magnitude > magnitudeAvg:
                ax.arrow(j, i, dx, dy, color='red')

    # Save the resulting image without displaying it
    plt.savefig(save_path)
    plt.close()




#spatial gradients are computed using the Sobel operator
#input: images
#output: first order derivatives

def get_first_order_derivatives(img1, img2):
    #derivative masks
    #Opted Kernal convolution to efficiently implement Fourier transformations
    x_kernel = np.array([[-1, 1], [-1, 1]]) * 0.25 #scaled to difference between neighbouring pixels in x direction
    y_kernel = np.array([[-1, -1], [1, 1]]) * 0.25
    t_kernel = np.ones((2, 2)) * 0.25

    fx = convolve(img1,x_kernel) + convolve(img2,x_kernel)
    fy = convolve(img1, y_kernel) + convolve(img2, y_kernel)
    ft = convolve(img1, -t_kernel) + convolve(img2, t_kernel)

    return [fx,fy, ft]


#input: first order derivatives
#output: second order derivatives
def get_second_Order_derivatives(fx, fy, ft):
    
    dxx_kernel = np.array([[1, -2, 1]])
    dyy_kernel = np.array([[1], [-2], [1]])
    dtt_kernel = np.array([[1, -2, 1]])


    fxx = convolve(fx, dxx_kernel)
    fyy = convolve(fy, dyy_kernel)
    ftt = convolve(ft, dtt_kernel)

    return [fxx,fyy, ftt]

#input: first order derivatives
#output: cross derivatives

def get_cross_derivatives(fx, fy, ft):
    # Define the cross derivative kernels
    dxy_kernel = np.array([[1, 1], [-1, -1]]) * 0.25
    dyt_kernel = np.array([[1], [1], [-1], [-1]]) * 0.25
    dtx_kernel = np.array([[1, -1], [1, -1]]) * 0.25

    # Calculate the cross derivatives
    dfxdy = convolve(fx, dxy_kernel)
    dfydt = convolve(fy, dyt_kernel)
    dftdx = convolve(ft, dtx_kernel)

    return [dfxdy, dfydt, dftdx]


#input: images name, smoothing parameter, tolerance
#output: images variations (flow vectors u, v)
#calculates u,v vectors and draw quiver
#alphta: smoothing parameter
#delta: tolerance

def optical_flow_estimation(name1, name2, delta):
    path = os.path.join(os.path.dirname(__file__), 'Primary Dataset/test images/')

    beforeImg = cv2.imread(os.path.join(path, name1), cv2.IMREAD_GRAYSCALE).astype(float)
    afterImg = cv2.imread(os.path.join(path, name2), cv2.IMREAD_GRAYSCALE).astype(float)


    if beforeImg is None:
        raise NameError("No input: \"" + name1 + '\"')
    elif afterImg is None:
        raise NameError("No input: \"" + name2 + '\"')

    

    #removing noise
    beforeImg  = cv2.GaussianBlur(beforeImg, (5, 5), 0)
    afterImg = cv2.GaussianBlur(afterImg, (5, 5), 0)

    # set up initial values
    #2-D numpy array of zeros with the same shape as beforeImg
    
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

       #of = fx*u_avg + fy*v_avg + ft

        #novel optical flow implementation
        of = (fx + 0.5*fxx + dfxdy) * u_avg + (fy + 0.5*fyy + dfydt) * v_avg + (ft + 0.5*ftt + dftdx)
        #d = 4 * alpha**2 + fx**2 + fy**2
        d = 1000 + fx**2 + fy**2


        prev = u

        u = u_avg - fx * (of / d)
        v = v_avg - fy * (of / d)

        diff = np.linalg.norm(u - prev, 2)
        #converges check (at most 300 iterations)
        if  diff < delta or iter_counter > 300:
            # print("iteration number: ", iter_counter)
            break

    quiver(u, v, beforeImg)

    return [u, v]



if __name__ == '__main__':
    #parser = ArgumentParser(description = 'OPTICAL FLOW ++')
    #parser.add_argument('img1', type = str, help = 'First image name (include format)')
    #parser.add_argument('img2', type = str, help='Second image name (include format)')
    #args = parser.parse_args()

    img1 = '/Users/chanukaalgama/Desktop/Optical_Flow++/OF_Test/test_images/car1.jpg'
    img2 = '/Users/chanukaalgama/Desktop/Optical_Flow++/OF_Test/test_images/car2.jpg'

    #image = cv2.imread(img2)
    #cv2.imwrite('ree.jpg', image)

    u,v = optical_flow_estimation(img1, img2, delta = 10**-1)
    end_time = time.time()
    execution_time = (end_time - start_time)/60
    print(f"Execution time: {execution_time} minutes")
