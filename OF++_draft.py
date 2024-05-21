import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage._filters import convolve
#to apply gausian laplace to n dimensional array
import os
from argparse import ArgumentParser



def show_image(name, image):
    if image is None:
        return

    cv2.imshow(name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



#compute magnitude in each 8 pixels. return magnitude average
def get_magnitude(u, v):
    scale = 3  # scaling factor - determines the sensitivity of flow vectors to motion
    sum = 0.0
    counter = 0.0

    for i in range(0, u.shape[0], 8):  #process every 8th element, reducing the total number of flow vectors considering efficiency
        for j in range(0, u.shape[1],8): 
            counter += 1
            dy = v[i,j] * scale
            dx = u[i,j] * scale
            # Euclidean distance formula - magnitude of flow vector at the current position
            magnitude = (dx**2 + dy**2)**0.5
            sum += magnitude

    mag_avg = sum / counter

    return mag_avg

#Quiver Plot

def draw_quiver(u,v,beforeImg):
    scale = 3
    ax = plt.figure().gca()
    ax.imshow(beforeImg, cmap = 'gray')

    magnitudeAvg = get_magnitude(u, v)

    for i in range(0, u.shape[0], 8):
        for j in range(0, u.shape[1],8):
            dy = v[i,j] * scale
            dx = u[i,j] * scale
            magnitude = (dx**2 + dy**2)**0.5
            #draw only significant changes
            if magnitude > magnitudeAvg:
                ax.arrow(j,i, dx, dy, color = 'red')

    plt.draw()
    plt.show()



#compute derivatives of the image intensity value changes (spatial gradient) along the x, y, time
#between the two consecutive frames

def get_first_Order_derivatives(img1, img2):
    #derivative masks
    #Opted Kernal convolution to efficiently implement Fourier transformations
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
def computeHS(name1, name2, alpha, delta):
    path = os.path.join(os.path.dirname(__file__), 'test images')
    beforeImg = cv2.imread(os.path.join(path, name1), cv2.IMREAD_GRAYSCALE)
    afterImg = cv2.imread(os.path.join(path, name2), cv2.IMREAD_GRAYSCALE)

    if beforeImg is None:
        raise NameError("Can't find image: \"" + name1 + '\"')
    elif afterImg is None:
        raise NameError("Can't find image: \"" + name2 + '\"')

    beforeImg = cv2.imread(os.path.join(path, name1), cv2.IMREAD_GRAYSCALE).astype(float)
    afterImg = cv2.imread(os.path.join(path, name2), cv2.IMREAD_GRAYSCALE).astype(float)

    #removing noise
    beforeImg  = cv2.GaussianBlur(beforeImg, (5, 5), 0)
    afterImg = cv2.GaussianBlur(afterImg, (5, 5), 0)


    # set up initial values
    u = np.zeros((beforeImg.shape[0], beforeImg.shape[1]))
    v = np.zeros((beforeImg.shape[0], beforeImg.shape[1]))
    fx, fy, ft = get_first_Order_derivatives(beforeImg, afterImg)
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
        d = 4 * alpha**2 + fx**2 + fy**2
        prev = u

        u = u_avg - fx * (p)
        v = v_avg - fy * (p)

        diff = np.linalg.norm(u - prev, 2)
        #converges check (at most 300 iterations)
        if  diff < delta or iter_counter > 300:
            # print("iteration number: ", iter_counter)
            break

    draw_quiver(u, v, beforeImg)

    return [u, v]



if __name__ == '__main__':
    parser = ArgumentParser(description = 'Horn Schunck program')
    parser.add_argument('img1', type = str, help = 'First image name (include format)')
    parser.add_argument('img2', type = str, help='Second image name (include format)')
    args = parser.parse_args()

    u,v = computeHS(args.img1, args.img2, alpha = 15, delta = 10**-1)






