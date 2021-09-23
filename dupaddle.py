from paddleocr import PaddleOCR, draw_ocr
from PIL import Image

def ocrProcess(img):
    ocr = PaddleOCR(use_angle_cls=True, lang='ch', det_limit_side_len=1280) # need to run only once to download and load model into memory

    result = ocr.ocr(img, cls=True)
    for line in result:
        print(line)

    image = Image.fromarray(img).convert('RGB')
    # boxes = result
    #
    # cvimg = np.array(image)
    # for box in boxes:
    #     box = np.reshape(np.array(box), [-1, 1, 2]).astype(np.int64)
    #     print(box)
    #     cvimg = cv2.polylines(np.array(cvimg), [box], True, (255, 0, 0), 2)
    #
    # im_show = Image.fromarray(cvimg)
    # im_show.save(r'D:\Dataset\ocr/result.jpg')
    # quit()

    boxes = [line[0] for line in result]
    txts = [line[1][0] for line in result]
    scores = [line[1][1] for line in result]
    im_show = draw_ocr(image, boxes, txts, scores, font_path='./fonts/simfang.ttf')
    im_show = Image.fromarray(im_show)
    im_show.save(r'D:\Dataset\ocr/result.jpg')

    return boxes, txts, scores



