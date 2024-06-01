import cv2
import numpy as np
import os

from PIL import Image
from predictor import Predictor

class Segment_Predict:
    def __init__(self):
        self.predictor = Predictor()
        self.result = []
    def segment_predict(self, path):
        self.image = cv2.imread(path)
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        lower_black = np.array([0,0,0])
        upper_black = np.array([200,255,54])
        mask = cv2.inRange(hsv, lower_black, upper_black)
        ret, binary = cv2.threshold(self.gray, 20, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_contour = sorted(contours, key = cv2.contourArea, reverse = True)[0]
        x, y, w, h = cv2.boundingRect(largest_contour)
        croped = self.image[y:y+h,x:x+w]
        _, binary = cv2.threshold(croped, 100, 255, cv2.THRESH_BINARY)
        kernel = np.uint8(np.zeros((3, 3)))
        for x in range(3):
            kernel[x, 2] = 1
            kernel[2, x] = 1
        eroded = cv2.erode(binary, kernel)
        binary_img = eroded[:, :, 0]
        js = []
        w = binary_img.shape[1]
        h = binary_img.shape[0]
        for x in range(w):
            j = 0
            for y in range(h):
                if binary_img[y][x] < 10:
                    j += 1
            js.append(j)

        positions = []
        for position, j in enumerate(js):
            if position < len(js) - 1:
                if j == 0 and js[position + 1] != 0:
                    positions.append(position)
        positions.append(w)
        for i in range(len(positions) - 1):
            seg = binary[:, positions[i]:positions[i + 1]]
            height, width, channel = seg.shape
            seg = seg[int(height/2)-200 : int(height/2)+200] 
            #print(height, width)
            cv2.imwrite('temp.jpg',seg)
            self.result.append(self.predictor.predict('temp.jpg', {0: '准', 1: '台', 2: '备', 3: '大', 4: '女', 5: '始', 6: '家', 7: '开'}))
            os.remove("temp.jpg")
            #cv2.imshow("Image", seg)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()
        return self.result

