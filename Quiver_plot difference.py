import cv2
import numpy as np
import matplotlib.pyplot as plt

def extract_image_difference(image1, image2):
    # Convert the images to grayscale
    image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Calculate the absolute difference between the images
    diff_image = cv2.absdiff(image1_gray, image2_gray)

    # Display the difference image
    plt.imshow(diff_image, cmap='gray')
    plt.axis('off')
    plt.show()

# Example usage
image1 = cv2.imread('test images/car1.jpg')
image2 = cv2.imread('test images/car2.jpg')

extract_image_difference(image1, image2)
