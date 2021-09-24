import os
from dupaddle import *
from dulines import *
from dudocx import *


def main(imgfn):
    img = cv2.imread(imgfn)

    lines_list = cannyLines(img)

    lines_list = mergeLines(lines_list, img)

    boxes, txts = ocrProcess(img)
    print(boxes)
    print(txts)
    docx_gen(lines_list, img, boxes, txts)


if __name__ == "__main__":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
    imgfn = r"D:\Dataset\ocr\0002.jpg"
    main(imgfn)
