import codecs
import cv2
import numpy as np
from PIL import Image
from collections import defaultdict
from ctypes import *


DEBUG = False

class Line(object):
    def __init__(self, line):
        if isinstance(line, np.ndarray):
            if len(line) == 2:
                self.p1 = line[0]
                self.p2 = line[1]
            elif len(line) == 4:
                self.p1 = line[0:2]
                self.p2 = line[2:]
        elif isinstance(line, list):
            self.p1 = np.array(line[0:2])
            self.p2 = np.array(line[2:])

        self.init()
        self.colstart = self.colend = 0
        self.rowstart = self.rowend = 0

    def __lt__(self, other):
        if self.vertical and not other.vertical:
            return False

        elif not self.vertical and other.vertical:
            return True

        elif self.vertical:
            if np.abs(self.p1[0] - other.p1[0]) > 10:
                if self.p1[0] < other.p1[0]:
                    return True
                else:
                    return False
            else:
                if self.p1[1] < other.p1[1]:
                    return True
                else:
                    return False
        else:
            if np.abs(self.p1[1] - other.p1[1]) > 10:
                if self.p1[1] < other.p1[1]:
                    return True
                else:
                    return False
            else:
                if self.p1[0] < other.p1[0]:
                    return True
                else:
                    return False

    def __repr__(self):
        return "P1: " + str(self.p1[0]) + " " + str(self.p1[1]) + " P2: " + str(self.p2[0]) + " " + str(self.p2[1])

    def init(self):
        self.valid = True

        self.angle = np.arctan((self.p2[1] - self.p1[1]) / (self.p2[0] - self.p1[0] + 0.0001)) * 180 / np.pi
        self.len = np.sqrt(np.square(self.p2[1] - self.p1[1]) + np.square(self.p2[0] - self.p1[0]))

        if np.abs(self.angle) > 45:
            self.vertical = True
            self.p = np.polyfit(np.array([self.p1[1], self.p2[1]]), np.array([self.p1[0], self.p2[0]]), 1)
            if self.p1[1] > self.p2[1]:
                self.p1, self.p2 = self.p2, self.p1
        else:
            self.vertical = False
            self.p = np.polyfit(np.array([self.p1[0], self.p2[0]]), np.array([self.p1[1], self.p2[1]]), 1)
            if self.p1[0] > self.p2[0]:
                self.p1, self.p2 = self.p2, self.p1

        self.center = ((self.p1[0] + self.p2[0]) * 0.5, (self.p1[1] + self.p2[1]) * 0.5)

    def is_same(self, line):
        global DEBUG
        if DEBUG:
            print(self.vertical, line.vertical)
            print(self.angle, line.angle)
            print(np.abs(self.angle - line.angle))
            print(line.center, self.center)
            print(line, self)
            print(self.overlap(line))
            print(self.point_dist(line.center))
            print(line.point_dist(self.center))
            DEBUG = False

        if self.vertical != line.vertical:
            return False

        if np.abs(self.angle - line.angle) > 15 and (180 - np.abs(self.angle - line.angle)) > 15:
            return False

        if self.point_dist(line.center) > 40:
            return False

        if line.point_dist(self.center) > 40:
            return False

        if not self.vertical:
            if np.abs(self.p1[0] - line.p2[0]) > 50 and np.abs(self.p2[0] - line.p1[0]) > 50 and not self.overlap(line):
                return False
        else:
            if np.abs(self.p1[1] - line.p2[1]) > 50 and np.abs(self.p2[1] - line.p1[1]) > 50 and not self.overlap(line):
                return False

        return True

    def point_dist(self, point):
        if not self.vertical:
            return np.abs(point[0] * self.p[0] - point[1] + self.p[1]) / np.sqrt(self.p[0] ** 2 + 1)
        else:
            return np.abs(self.p[0] * point[1] - point[0] + self.p[1]) / np.sqrt(self.p[0] ** 2 + 1)

    def overlap(self, line):
        if not self.vertical:
            if max(self.p1[0], line.p1[0]) >= (min(self.p2[0], line.p2[0]) + 10):
                return False
        else:
            if max(self.p1[1], line.p1[1]) >= (min(self.p2[1], line.p2[1]) + 10):
                return False

        return True

    def merge(self, line):
        if not self.vertical:
            if self.p1[0] > line.p1[0]:
                self.p1 = line.p1

            if self.p2[0] < line.p2[0]:
                self.p2 = line.p2
        else:
            if self.p1[1] > line.p1[1]:
                self.p1 = line.p1

            if self.p2[1] < line.p2[1]:
                self.p2 = line.p2

        self.init()

    def is_frame(self, w, h, bound_ratio=0.03):
        top_bound = h * bound_ratio
        bottom_bound = h * (1 - bound_ratio)
        left_bound = w * bound_ratio
        right_bound = w * (1 - bound_ratio)
        print(self.vertical, self.center, left_bound, right_bound, top_bound, bottom_bound)
        if self.vertical and (self.center[0] < left_bound or self.center[0] > right_bound):
            return True

        if not self.vertical and (self.center[1] < top_bound or self.center[1] > bottom_bound):
            return True

        return False

    def in_bound(self, bound):
        bound_thresh = 10
        top, left, bottom, right = bound

        if not self.vertical:
            if (top < self.center[1] < bottom) and ((left - bound_thresh) <
                    self.p1[0] < (right + bound_thresh)) and ((left - bound_thresh) <
                    self.p2[0] < (right + bound_thresh)):
                return True
        else:
            if (left < self.center[0] < right) and ((top - bound_thresh) <
                    self.p1[0] < (bottom + bound_thresh)) and ((top - bound_thresh) <
                    self.p2[0] < (bottom + bound_thresh)):
                return True

        return False

def invalid_code(textfn=r"D:\Dataset\ocr\0006.txt"):
    # with codecs.open(textfn, mode="r", encoding="utf-8") as f:
    #     text_data = f.readlines()

    # boxes = []
    # texts = []
    # scores = []
    # for data in text_data:
    #     data = data.strip().split(";")
    #     box = data[0].split()
    #     box = list(map(float, box))
    #     box = np.array(box)
    #     box = np.reshape(box, (-1, 2)).astype(np.int32)
    #     text = data[1]
    #     score = float(data[2])
    #
    #     boxes.append(box)
    #     texts.append(text)
    #     scores.append(score)

    # for box in boxes:
    #     cv2.line(img, tuple(box[0]), tuple(box[1]), (0, 255, 0), 5)
    #     cv2.line(img, tuple(box[1]), tuple(box[2]), (0, 255, 0), 5)
    #     cv2.line(img, tuple(box[2]), tuple(box[3]), (0, 255, 0), 5)
    #     cv2.line(img, tuple(box[3]), tuple(box[0]), (0, 255, 0), 5)
    pass


def mergeLines(lines, src):
    global DEBUG

    img = src.copy()
    imgh, imgw = img.shape[:2]

    lines.sort()

    img2 = img.copy()
    for line in lines:
        # if not line.vertical:
        #     continue
        #print(line.p1, line.p2)
        cv2.line(img2, tuple(line.p1.astype(np.int32)), tuple(line.p2.astype(np.int32)), (0, 255, 0), 5)
    print("**********************")
    cv2.namedWindow("img2", 0)
    cv2.imshow("img2", img2)
    cv2.waitKey(2)

    for li, line in enumerate(lines):
        print(li, line)

    for i in range(len(lines)):
        if not lines[i].valid:
            continue

        for j in range(len(lines)):
            if not lines[j].valid:
                continue

            if j == i:
                continue

            if i == 18 and j == 19:
                DEBUG = False
            if lines[i].is_same(lines[j]):
                lines[i].merge(lines[j])
                lines[j].valid = False
                if DEBUG:
                    print("merged", i, j)
                    print(i, lines[i])
                    print(j, lines[j])
            # cv2.imshow("img3", img3)
            # cv2.waitKey(3)

        print("***********", i)
        # if i > 4:
        #     quit()

    merged_lines = []
    for i in range(len(lines)):
        if not lines[i].valid:
            continue

        if lines[i].is_frame(imgw, imgh):
            continue

        merged_lines.append(lines[i])

    if len(merged_lines) > 0:
        step = 255 // len(merged_lines)
        for li, line in enumerate(merged_lines):
            cv2.line(img, tuple(line.p1.astype(np.int32)), tuple(line.p2.astype(np.int32)), (0, 255 - step * li, step * li), 5)

    cv2.namedWindow("img", 0)
    cv2.imshow("img", img)
    cv2.waitKey()

    with open(textfn.replace(".txt", "_result.txt"), "w") as f:
        for li, line in enumerate(merged_lines):
            f.write(str(line.p1[0]) + " " + str(line.p1[1]) + " " + str(line.p2[0]) + " " + str(line.p2[1]) + "\n")

    return merged_lines


linefn, imgfn, textfn = r"D:\Dataset\ocr\0002_lines.txt", r"D:\Dataset\ocr\0003.jpg", r"D:\Dataset\ocr\0002.txt"
#mergeLines(r"D:\Dataset\ocr\0003_lines.txt", r"D:\Dataset\ocr\0003.jpg", r"D:\Dataset\ocr\0003.txt")
#quit()


pDll = CDLL(r"D:\Program\Project\ncnn-master\paddle_ocr\CannyLines_dll.dll")
def cannyLines(img):
    global pDll

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    c_uchar_p = POINTER(c_uint8)
    data_p = img.ctypes.data_as(c_uchar_p)

    lines_arr = np.zeros((4096 * 4), dtype=np.float32)
    c_float_p = POINTER(c_float)
    lines_p = lines_arr.ctypes.data_as(c_float_p)

    line_num = c_int32()
    ret = pDll.cannyLineDetect(data_p, img.shape[1], img.shape[0], lines_p, byref(line_num))
    print(line_num, ret)

    lines_list = []
    for i in range(line_num.value):
        lines_list.append(Line(lines_arr[i * 4:(i + 1) * 4]))

    return lines_list


def read_line_file(linefn):
    with open(linefn) as f:
        datas = f.readlines()
        lines = []
        for data in datas:
            data = data.strip().split(" ")
            data = list(map(float, data))
            lines.append(Line(data))

    return lines


def Test():
    test_units = [(r"D:\Dataset\ocr\0001_lines.txt", r"D:\Dataset\ocr\0001.jpg", 0),
                  (r"D:\Dataset\ocr\0002_lines.txt", r"D:\Dataset\ocr\0002.jpg", 34),
                  (r"D:\Dataset\ocr\0003_lines.txt", r"D:\Dataset\ocr\0003.jpg", 24),
                  (r"D:\Dataset\ocr\0004_lines.txt", r"D:\Dataset\ocr\0004.jpg", 22),
                  (r"D:\Dataset\ocr\0005_lines.txt", r"D:\Dataset\ocr\0005.jpg", 4),
                  (r"D:\Dataset\ocr\0006_lines.txt", r"D:\Dataset\ocr\0006.jpg", 4),
                  (r"D:\Dataset\ocr\0007_lines.txt", r"D:\Dataset\ocr\0007.jpg", 4),
                  (r"D:\Dataset\ocr\0008_lines.txt", r"D:\Dataset\ocr\0008.jpg", 4),
                  (r"D:\Dataset\ocr\0009_lines.txt", r"D:\Dataset\ocr\0009.jpg", 4),
                  (r"D:\Dataset\ocr\0010_lines.txt", r"D:\Dataset\ocr\0010.jpg", 4),
                  (r"D:\Dataset\ocr\0011_lines.txt", r"D:\Dataset\ocr\0011.jpg", 12)]

    for test_uint in test_units:
        img = cv2.imread(test_uint[1])
        lines = read_line_file(test_uint[0])
        lines = mergeLines(lines, img)

        if len(lines) != test_uint[2]:
            print("Test failed", test_uint[1])
            quit()

    quit()


if __name__ == "__main__":
    #Test()

    img = cv2.imread(r"D:\Dataset\ocr\0002.jpg")
    lines_list = cannyLines(img)

    lines = mergeLines(lines_list, img)
    print(len(lines))
