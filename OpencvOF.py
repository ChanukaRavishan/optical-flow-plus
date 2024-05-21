import cv2
import numpy as np
import matplotlib.pyplot as plt

def calculate_optical_flow(image1, image2):
    # Convert the images to grayscale
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Calculate dense optical flow
    flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, pyr_scale=0.5, levels=5, winsize=11, iterations=5,
                                        poly_n=5, poly_sigma=1.1, flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN)

    # Separate the flow into u and v components
    u = flow[:, :, 0]
    v = flow[:, :, 1]

    # Calculate the magnitude of flow vectors
    magnitude = np.sqrt(u ** 2 + v ** 2)

    # Calculate the average magnitude of flow vectors
    avg_magnitude = np.mean(magnitude)
    
    ax = plt.figure().gca()
    ax.imshow(image1, cmap = 'gray')

    for i in range(0, u.shape[0], 8):
        for j in range(0, u.shape[1],8):
            dy = v[i,j]
            dx = u[i,j]
            magnitude = (dx**2 + dy**2)**0.5
            #draw only significant changes
            if magnitude > avg_magnitude:
                ax.arrow(j,i, dx, dy, color = 'red')

    plt.draw()
    plt.show()
    
    
    # # Create a quiver plot to visualize the flow vectors
    # fig, ax = plt.subplots()
    # ax.quiver(u, v, color='red')
    # ax.imshow(image1, cmap='gray')
    # plt.axis('off')
    # plt.show()

# Example usage
image1 = cv2.imread('test images/car1.jpg')
image2 = cv2.imread('test images/car2.jpg')

calculate_optical_flow(image1, image2)



