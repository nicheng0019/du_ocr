from dulines import *
from dudocx import *
from dupaddle import *


def main(imgfn):
    img = cv2.imread(imgfn)

    lines_list = cannyLines(img)

    lines_list = mergeLines(lines_list, img)

    boxes, txts, scores = ocrProcess(img)



if __name__ == "__main__":
    imgfn = r""
    main(imgfn)