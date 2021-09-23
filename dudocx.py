import cv2
from dulines import *


def docx_demo(linefn, imgfn, textfn):
    from docx import Document
    from docx.shared import Inches, Cm

    paper_height = 29.7
    paper_width = 21

    img = cv2.imread(imgfn)
    imgh, imgw = img.shape[:2]

    with open(linefn) as f:
        datas = f.readlines()

    lines = []
    lines_arr = []
    for data in datas:
        data = data.strip().split()
        data = list(map(float, data))
        line = Line(data)
        lines.append(line)
        lines_arr.append([line.p1, line.p2])

    avgangle = 0
    count = 0
    for line in lines:
        if not line.vertical:
            avgangle += line.angle
            count += 1

    if count == 0:
        for line in lines:
            if line.vertical:
                if line.angle > 0:
                    avgangle += line.angle - 90
                else:
                    avgangle += line.angle + 90
                count += 1

    assert count != 0
    avgangle = avgangle / count
    print("avgangle: ", avgangle)

    center = (imgw / 2, imgh / 2)
    M = cv2.getRotationMatrix2D(center, avgangle, 1)
    lines_arr = np.array(lines_arr)
    lines_arr = lines_arr.dot(M[:, :2].T) + M[:, 2]

    img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

    lines = []
    for line in lines_arr:
        lines.append(Line(line))

    lines.sort()

    def get_top_line(lines):
        ymin = imgh
        topidx = -1
        for li, line in enumerate(lines):
            if not line.valid:
                continue

            if line.vertical:
                continue

            if line.center[1] < ymin:
                ymin = line.center[1]
                topidx = li

        return topidx

    def points_dist(p1, p2):
        return np.linalg.norm(p1 - p2)

    def get_rect(lines, topidx):
        topline = lines[topidx]
        topp1 = topline.p1
        topp2 = topline.p2

        rects = []

        linenum = len(lines)
        for leftidx in range(linenum):
            leftline = lines[leftidx]
            if not leftline.valid:
                continue

            if not leftline.vertical:
                continue

            leftp1 = leftline.p1
            leftp2 = leftline.p2

            if points_dist(leftp1, topp1) > 20:
                continue

            for bottomidx in range(linenum):
                bottomline = lines[bottomidx]
                if not bottomline.valid:
                    continue

                if bottomline.vertical:
                    continue

                if bottomidx == topidx:
                    continue

                bottomp1 = bottomline.p1
                bottomp2 = bottomline.p2

                if points_dist(leftp2, bottomp1) > 20:
                    continue

                for rightidx in range(linenum):
                    rightline = lines[rightidx]
                    if not rightline.valid:
                        continue

                    if not rightline.vertical:
                        continue

                    if leftidx == rightidx:
                        continue

                    rightp1 = rightline.p1
                    rightp2 = rightline.p2

                    if points_dist(rightp2, bottomp2) > 20:
                        continue

                    if points_dist(rightp1, topp2) > 20:
                        continue

                    rects.append([topidx, leftidx, bottomidx, rightidx])

        return rects

    def choose_rect(lines, rects):
        maxarea = 0
        maxidx = -1
        for ri, rect in enumerate(rects):
            top, left, bottom, right = rect
            topline, leftline, bottomline, rightline = lines[top], lines[left], lines[bottom], lines[right]
            area = (bottomline.center[1] - topline.center[1]) * (rightline.center[0] - leftline.center[0])

            if area > maxarea:
                maxarea = area
                maxidx = ri

        return rects[maxidx]

    def get_all_lines_in_rect(lines, rect):
        top, left, bottom, right = rect
        topline, leftline, bottomline, rightline = lines[top], lines[left], lines[bottom], lines[right]

        bound = [topline.center[1], leftline.center[0], bottomline.center[1], rightline.center[0]]
        sub_lines = []
        sub_lines.extend(rect)
        for li, line in enumerate(lines):
            if not line.valid:
                continue

            if line.in_bound(bound):
                sub_lines.append(li)

        return sub_lines

    def get_sub_table(lines, sub_lines):
        THRESH = 10
        topline, leftline, bottomline, rightline = lines[sub_lines[0]],\
                                lines[sub_lines[1]], lines[sub_lines[2]], lines[sub_lines[3]]

        horizontal_lines_id = []
        vertical_lines_id = []
        for li in sub_lines[4:]:
            if lines[li].vertical:
                vertical_lines_id.append(li)
            else:
                horizontal_lines_id.append(li)

        row_num = column_num = 1

        prevrow = topline.center[1]
        row_ys = defaultdict(list)
        row_ys[row_num - 1].append(topline.center[1])

        for li in horizontal_lines_id:
            line = lines[li]
            if line.center[1] - prevrow > THRESH:
                row_num += 1

            lines[li].rowstart = lines[li].rowend = row_num - 1
            prevrow = line.center[1]
            row_ys[row_num - 1].append(line.center[1])

        prevcol = leftline.center[0]
        col_xs = defaultdict(list)
        col_xs[column_num - 1].append(leftline.center[0])

        for li in vertical_lines_id:
            line = lines[li]
            if line.center[0] - prevcol > THRESH:
                column_num += 1

            lines[li].colstart = lines[li].colend = column_num - 1
            prevcol = line.center[0]
            col_xs[column_num - 1].append(line.center[0])

        row_num += 1
        column_num += 1
        print(row_num, column_num)

        lines[sub_lines[0]].rowstart = lines[sub_lines[0]].rowend = 0
        lines[sub_lines[2]].rowstart = lines[sub_lines[2]].rowend = row_num - 1

        lines[sub_lines[0]].colstart = lines[sub_lines[2]].colstart = 0
        lines[sub_lines[0]].colend = lines[sub_lines[2]].colend = column_num - 1

        lines[sub_lines[1]].colstart = lines[sub_lines[1]].colend = 0
        lines[sub_lines[3]].colstart = lines[sub_lines[3]].colend = column_num - 1

        lines[sub_lines[1]].rowstart = lines[sub_lines[3]].rowstart = 0
        lines[sub_lines[1]].rowend = lines[sub_lines[3]].rowend = row_num - 1

    while True:
        topidx = get_top_line(lines)
        if -1 == topidx:
            break

        rects = get_rect(lines, topidx)
        for rect in rects:
            cv2.line(img, tuple(lines[rect[0]].p1.astype(np.int32)), tuple(lines[rect[0]].p2.astype(np.int32)), (0, 255, 0), 10)
            cv2.line(img, tuple(lines[rect[1]].p1.astype(np.int32)), tuple(lines[rect[1]].p2.astype(np.int32)), (0, 255, 0), 10)
            cv2.line(img, tuple(lines[rect[2]].p1.astype(np.int32)), tuple(lines[rect[2]].p2.astype(np.int32)), (0, 255, 0), 10)
            cv2.line(img, tuple(lines[rect[3]].p1.astype(np.int32)), tuple(lines[rect[3]].p2.astype(np.int32)), (0, 255, 0), 10)

            lines[rect[0]].valid = False
            lines[rect[1]].valid = False
            lines[rect[2]].valid = False
            lines[rect[3]].valid = False

        rect = choose_rect(lines, rects)

        sub_lines = get_all_lines_in_rect(lines, rect)
        for li in sub_lines[4:]:
            lines[li].valid = False
            cv2.line(img, tuple(lines[li].p1.astype(np.int32)), tuple(lines[li].p2.astype(np.int32)),
                     (255, 0, 0), 10)

        cv2.namedWindow("img", 0)
        cv2.imshow("img", img)
        cv2.waitKey()

        get_sub_table(lines, sub_lines)

    quit()

    document = Document()

    records = (
        (3, '101', 'Spam'),
        (7, '422', 'Eggs'),
        (4, '631', 'Spam, spam, eggs, and spam')
    )

    table = document.add_table(rows=1, cols=3,  style='TableGrid')
    hdr_cells = table.rows[0].cells
    table.rows[0].height = Cm(0.7)
    hdr_cells[0].text = 'Qty'
    hdr_cells[0].width = Cm(5.0)
    hdr_cells[1].text = 'Id'
    hdr_cells[1].width = Cm(5.0)
    hdr_cells[2].text = 'Desc'
    hdr_cells[2].width = Cm(5.0)
    for i, (qty, id, desc) in enumerate(records):
        row_cells = table.add_row().cells
        table.rows[i + 1].height = Cm(0.7)
        row_cells[0].text = str(qty)
        row_cells[0].width = Cm(5.0)
        row_cells[1].text = id
        row_cells[1].width = Cm(5.0)
        row_cells[2].text = desc
        row_cells[2].width = Cm(5.0)

    document.save('d:/demo.docx')


docx_demo(linefn, imgfn, textfn)
