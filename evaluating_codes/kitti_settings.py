import numpy as np
from PIL import Image

class FlowImage:

    def __init__(self):
        self.data_ = None
        self.width_ = 0
        self.height_ = 0

    def readFlowField(self, file_name):
        image = Image.open(file_name)
        width, height = image.size
        self.width_ = width
        self.height_ = height
        self.data_ = np.zeros((width * height * 3,), dtype=np.float32)

        for v in range(height):
            for u in range(width):
                val = image.getpixel((u, v))
                if val[2] > 0:
                    self.setFlowU(u, v, (val[0] - 32768.0) / 64.0)
                    self.setFlowV(u, v, (val[1] - 32768.0) / 64.0)
                    self.setValid(u, v, True)
                else:
                    self.setFlowU(u, v, 0)
                    self.setFlowV(u, v, 0)
                    self.setValid(u, v, False)

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

    def interpolateBackground(self):
        for v in range(self.height_):
            count = 0
            for u in range(self.width_):
                if self.isValid(u, v):
                    if count >= 1:
                        u1 = u - count
                        u2 = u - 1
                        if 0 < u1 < self.width_ - 1 and 0 < u2 < self.width_ - 1:
                            fu_ipol = min(self.getFlowU(u1 - 1, v), self.getFlowU(u2 + 1, v))
                            fv_ipol = min(self.getFlowV(u1 - 1, v), self.getFlowV(u2 + 1, v))
                            for u_curr in range(u1, u2 + 1):
                                self.setFlowU(u_curr, v, fu_ipol)
                                self.setFlowV(u_curr, v, fv_ipol)
                                self.setValid(u_curr, v, True)
                    count = 0
                else:
                    count += 1

            for u in range(self.width_):
                if self.isValid(u, v):
                    for u2 in range(u):
                        self.setFlowU(u2, v, self.getFlowU(u, v))
                        self.setFlowV(u2, v, self.getFlowV(u, v))
                        self.setValid(u2, v, True)
                    break

            for u in range(self.width_ - 1, -1, -1):
                if self.isValid(u, v):
                    for u2 in range(u + 1, self.width_):
                        self.setFlowU(u2, v, self.getFlowU(u, v))
                        self.setFlowV(u2, v, self.getFlowV(u, v))
                        self.setValid(u2, v, True)
                    break

        for u in range(self.width_):
            for v in range(self.height_):
                if self.isValid(u, v):
                    for v2 in range(v):
                        self.setFlowU(u, v2, self.getFlowU(u, v))
                        self.setFlowV(u, v2, self.getFlowV(u, v))
                        self.setValid(u, v2, True)
                    break

            for v in range(self.height_ - 1, -1, -1):
                if self.isValid(u, v):
                    for v2 in range(v + 1, self.height_):
                        self.setFlowU(u, v2, self.getFlowU(u, v))
                        self.setFlowV(u, v2, self.getFlowV(u, v))
                        self.setValid(u, v2, True)
                    break

    def errorImage(self, F_noc, F_occ, log_colors=False):
        n = 8
        image = Image.new('RGB', (self.width_, self.height_))
        for v in range(1, self.height_ - 1):
            for u in range(1, self.width_ - 1):
                if F_occ.isValid(u, v):
                    val = (0, 0, 0)
                    if log_colors:
                        dfu = self.getFlowU(u, v) - F_occ.getFlowU(u, v)
                        dfv = self.getFlowV(u, v) - F_occ.getFlowV(u, v)
                        f_err = np.sqrt(dfu * dfu + dfv * dfv)
                        f_mag = F_occ.getFlowMagnitude(u, v)
                        n_err = min(f_err / 3.0, 20.0 * f_err / f_mag)
                        for i in range(10):
                            if LC[i][0] <= n_err < LC[i][1]:
                                val = (int(LC[i][2]), int(LC[i][3]), int(LC[i][4]))
                        if not F_noc.isValid(u, v):
                            val = (int(val[0] * 0.5), int(val[1] * 0.5), int(val[2] * 0.5))
                    else:
                        dfu = self.getFlowU(u, v) - F_occ.getFlowU(u, v)
                        dfv = self.getFlowV(u, v) - F_occ.getFlowV(u, v)
                        f_err = min(np.sqrt(dfu * dfu + dfv * dfv), 5.0) / 5.0
                        val = (int(f_err * 255.0), int(f_err * 255.0), int(f_err * 255.0))
                        if not F_noc.isValid(u, v):
                            val = (0, 0, 0)
                    for v2 in range(v - 1, v + 2):
                        for u2 in range(u - 1, u + 2):
                            image.putpixel((u2, v2), val)
        return image

    def hsvToRgb(self, h, s, v):
        c = v * s
        h2 = 6.0 * h
        x = c * (1.0 - abs(h2 % 2.0 - 1.0))
        if 0 <= h2 < 1:
            return c, x, 0
        elif 1 <= h2 < 2:
            return x, c, 0
        elif 2 <= h2 < 3:
            return 0, c, x
        elif 3 <= h2 < 4:
            return 0, x, c
        elif 4 <= h2 < 5:
            return x, 0, c
        elif 5 <= h2 <= 6:
            return c, 0, x
        elif h2 > 6:
            return 1, 0, 0
        elif h2 < 0:
            return 0, 1, 0

    def writeFalseColors(self, file_name, max_flow):
        n = 8
        image = Image.new('RGB', (self.width_, self.height_))
        for v in range(self.height_):
            for u in range(self.width_):
                r, g, b = 0, 0, 0
                if self.isValid(u, v):
                    mag = self.getFlowMagnitude(u, v)
                    direction = np.arctan2(self.getFlowV(u, v), self.getFlowU(u, v))
                    h = (direction / (2.0 * np.pi) + 1.0) % 1.0
                    s = min(max(mag * n / max_flow, 0.0), 1.0)
                    v = min(max(n - s, 0.0), 1.0)
                    r, g, b = self.hsvToRgb(h, s, v)
                image.putpixel((u, v), (int(r * 255.0), int(g * 255.0), int(b * 255.0)))
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

    def maxFlow(self):
        max_flow = 0
        for u in range(self.width_):
            for v in range(self.height_):
                if self.isValid(u, v) and self.getFlowMagnitude(u, v) > max_flow:
                    max_flow = self.getFlowMagnitude(u, v)
        return max_flow


LC = [
    [0.000000, 0.062500, 0.375, 0.000, 0.000],
    [0.062500, 0.125000, 0.625, 0.000, 0.000],
    [0.125000, 0.250000, 0.750, 1.000, 1.000],
    [0.250000, 0.375000, 0.875, 1.000, 0.000],
    [0.375000, 0.500000, 1.000, 1.000, 0.000],
    [0.500000, 0.625000, 1.000, 0.000, 0.000],
    [0.625000, 0.750000, 1.000, 0.000, 1.000],
    [0.750000, 0.875000, 1.000, 1.000, 0.000],
    [0.875000, 1.000000, 0.750, 1.000, 0.000],
    [1.000000, 1.200000, 0.500, 0.000, 0.000]
]

# Example usage:
flow_image = FlowImage()
flow_image.readFlowField("KITTI Dataset/training/flow_occ//000046_10.png")
flow_image.interpolateBackground()
flow_image.writeFlowField("interpolated_flow_field.png")
# max_flow = flow_image.maxFlow()
# flow_image.writeFalseColors("path/to/false_colors.png", max_flow)
# error_image = flow_image.errorImage(FlowImage(), FlowImage())
# error_image.save("path/to/error_image.png")
