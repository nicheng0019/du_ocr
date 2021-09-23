from dulines import *


def main(imgfn):
    img = cv2.imread(imgfn)

    lines_list = cannyLines(img)

    lines_list = mergeLines(lines_list, img)



if __name__ == "__main__":
    imgfn = r""
    main(imgfn)