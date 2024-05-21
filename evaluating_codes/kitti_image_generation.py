import cv2
import numpy as np
from PIL import Image


class FlowImage:

    def __init__(self):
        self.data_ = None
        self.width_ = 0
        self.height_ = 0

    def readFlowField(self, file_name1, file_name2):
        image1 = Image.open(file_name1)
        #image2 = Image.open(file_name2)
        width, height = image1.size
        self.width_ = width
        self.height_ = height
        self.data_ = np.zeros((width * height * 3,), dtype=np.float32)
        #flow = np.zeros((width * height * 3,), dtype=np.float32)

        
        image1 = cv2.imread(file_name1)
        image2 = cv2.imread(file_name2)
        
        gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, pyr_scale=0.5, levels=5, winsize=11, iterations=5,
                                         poly_n=5, poly_sigma=1.1, flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN)

        opencv_u = flow[:, :, 0]
        opencv_v = flow[:, :, 1]

        for v in range(height):
            for u in range(width):
                val_u = opencv_u[v, u]
                val_v = opencv_v[v, u]
                self.setFlowU(u, v, (val_u - 32768.0) / 64.0)
                self.setFlowV(u, v, (val_v - 32768.0) / 64.0)
                self.setValid(u, v, True)

    def writeFlowField(self, file_name):
        image = Image.new('RGB', (self.width_, self.height_))
        for v in range(self.height_):
            for u in range(self.width_):
                val = [0, 0, 0]
                if self.isValid(u, v):
                    val[0] = int(max(min(self.getFlowU(u, v) * 64.0 + 32768.0, 65535.0), 0.0))
                    val[1] = int(max(min(self.getFlowV(u, v) * 64.0 + 32768.0, 65535.0), 0.0))
                    val[2] = 1
                image.putpixel((u, v), tuple(val))
        image.save(file_name)



    def getFlowU(self, u, v):
        return self.data_[3 * (v * self.width_ + u) + 0]

    def getFlowV(self, u, v):
        return self.data_[3 * (v * self.width_ + u) + 1]

    def isValid(self, u, v):
        return self.data_[3 * (v * self.width_ + u) + 2] > 0.5

    def getFlowMagnitude(self, u, v):
        fu = self.getFlowU(u, v)
        fv = self.getFlowV(u, v)
        return np.sqrt(fu * fu + fv * fv)

    def setFlowU(self, u, v, val):
        self.data_[3 * (v * self.width_ + u) + 0] = val

    def setFlowV(self, u, v, val):
        self.data_[3 * (v * self.width_ + u) + 1] = val

    def setValid(self, u, v, valid):
        self.data_[3 * (v * self.width_ + u) + 2] = 1 if valid else 0



sequence_folder = 'KITTI Dataset/training/image_3'
ground_truth_folder = 'KITTI Dataset/training/flow_occ'
image1 = sequence_folder + '/000006_10.png'
image2 = sequence_folder + '/000006_11.png'
ground_truth_flow = cv2.imread(ground_truth_folder + '/000006_10.png')
output = 'Results/xxxx.png'


flow_image = FlowImage()
flow_image.readFlowField(image1, image2)
flow_image.writeFlowField(output)
