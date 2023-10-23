import glob

from numpy import expand_dims
from keras.models import load_model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras import backend as K

import cv2
import numpy as np

import os
import argparse


class BoundBox:
    def __init__(self, xmin, ymin, xmax, ymax, objness = None, classes = None):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.objness = objness
        self.classes = classes
        self.label = -1
        self.score = -1
        
    def get_label(self):
        if self.label == -1:
            self.label = np.argmax(self.classes)
        
        return self.label
    
    def get_score(self):
        if self.score == -1:
            self.score = self.classes[self.get_label()]
        
        return self.score


def _sigmoid(x):
    return 1. / (1. + np.exp(-x))
    

def decode_netout(netout, anchors, obj_thresh, net_h, net_w):
    grid_h, grid_w = netout.shape[:2] # 0 and 1 is row and column 13*13
    nb_box = 3 # 3 anchor boxes
    netout = netout.reshape((grid_h, grid_w, nb_box, -1)) #13*13*3 ,-1
    nb_class = netout.shape[-1] - 5
    boxes = []
    netout[..., :2]  = _sigmoid(netout[..., :2])
    netout[..., 4:]  = _sigmoid(netout[..., 4:])
    netout[..., 5:]  = netout[..., 4][..., np.newaxis] * netout[..., 5:]
    netout[..., 5:] *= netout[..., 5:] > obj_thresh
    
    for i in range(grid_h*grid_w):
        row = i / grid_w
        col = i % grid_w
        for b in range(nb_box):
            # 4th element is objectness score
            objectness = netout[int(row)][int(col)][b][4]
            if(objectness.all() <= obj_thresh): continue
            # first 4 elements are x, y, w, and h
            x, y, w, h = netout[int(row)][int(col)][b][:4]
            x = (col + x) / grid_w # center position, unit: image width
            y = (row + y) / grid_h # center position, unit: image height
            w = anchors[2 * b + 0] * np.exp(w) / net_w # unit: image width
            h = anchors[2 * b + 1] * np.exp(h) / net_h # unit: image height
            # last elements are class probabilities
            classes = netout[int(row)][col][b][5:]
            box = BoundBox(x-w/2, y-h/2, x+w/2, y+h/2, objectness, classes)
            boxes.append(box)
    return boxes


def correct_yolo_boxes(boxes, image_h, image_w, net_h, net_w):
    new_w, new_h = net_w, net_h
    for i in range(len(boxes)):
        x_offset, x_scale = (net_w - new_w)/2./net_w, float(new_w)/net_w
        y_offset, y_scale = (net_h - new_h)/2./net_h, float(new_h)/net_h
        boxes[i].xmin = int((boxes[i].xmin - x_offset) / x_scale * image_w)
        boxes[i].xmax = int((boxes[i].xmax - x_offset) / x_scale * image_w)
        boxes[i].ymin = int((boxes[i].ymin - y_offset) / y_scale * image_h)
        boxes[i].ymax = int((boxes[i].ymax - y_offset) / y_scale * image_h)


def _interval_overlap(interval_a, interval_b):
    x1, x2 = interval_a
    x3, x4 = interval_b
    if x3 < x1:
        if x4 < x1:
            return 0
        else:
            return min(x2,x4) - x1
    else:
        if x2 < x3:
            return 0
        else:
            return min(x2,x4) - x3



def bbox_iou(box1, box2):
    intersect_w = _interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
    intersect_h = _interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])
    intersect = intersect_w * intersect_h

    w1, h1 = box1.xmax-box1.xmin, box1.ymax-box1.ymin  
    w2, h2 = box2.xmax-box2.xmin, box2.ymax-box2.ymin
    
    #Union(A,B) = A + B - Inter(A,B)
    union = w1*h1 + w2*h2 - intersect
    return float(intersect) / union


def do_nms(boxes, nms_thresh):    #boxes from correct_yolo_boxes and  decode_netout
    if len(boxes) > 0:
        nb_class = len(boxes[0].classes)
    else:
        return
    for c in range(nb_class):
        sorted_indices = np.argsort([-box.classes[c] for box in boxes])
        for i in range(len(sorted_indices)):
            index_i = sorted_indices[i]
            if boxes[index_i].classes[c] == 0: continue
            for j in range(i+1, len(sorted_indices)):
                index_j = sorted_indices[j]
                if bbox_iou(boxes[index_i], boxes[index_j]) >= nms_thresh:
                    boxes[index_j].classes[c] = 0


# load and prepare an image
def load_image_pixels(filename, shape):
    # load the image to get its shape
    image = load_img(filename) #load_img() Keras function to load the image .
    width, height = image.size
    # load the image with the required size
    image = load_img(filename, target_size=shape) # target_size argument to resize the image after loading
    # convert to numpy array
    image = img_to_array(image)
    # scale pixel values to [0, 1]
    image = image.astype('float32')
    image /= 255.0  # rescale the pixel values from 0-255 to 0-1 32-bit floating point values.
    # add a dimension so that we have one sample
    image = expand_dims(image, 0)
    return image, width, height


# get all of the results above a threshold
def get_boxes(boxes, labels, thresh):
    v_boxes, v_labels, v_scores = list(), list(), list()
    # enumerate all boxes
    for box in boxes:
        # enumerate all possible labels
        for i in range(len(labels)):
            # check if the threshold for this label is high enough
            if box.classes[i] > thresh:
                v_boxes.append(box)
                v_labels.append(labels[i])
                v_scores.append(box.classes[i]*100)
    
    return v_boxes, v_labels, v_scores


# draw all results
def draw_boxes(filename, v_boxes, v_labels, v_scores, output_dir, size_reshape = (96, 96)):
    output = output_dir + '_yolov3.jpg'
    #load the image
    img = cv2.imread(filename)
    img_rs = cv2.resize(img, size_reshape, cv2.INTER_AREA)
    count_obj = 0
    # img_out = None
    str_result = "/".join(filename.rsplit("/")[4:])
    # with open(output_dir + '/label_img_yolo' + '.txt', 'w') as f:
    #     f.write(info_image_result)

    if len(v_boxes) == 0:
        return None
    else:
        str_result += f" [{len(v_boxes)}]"

    for ind, i in enumerate(range(len(v_boxes))):
        count_obj += 1
        # retrieving the coordinates from each bounding box
        label = v_labels[i]
        box = v_boxes[i]
        y1, x1, y2, x2 = box.ymin, box.xmin, box.ymax, box.xmax
        y1 = int(y1 * size_reshape[0] / img.shape[0])
        y2 = int(y2 * size_reshape[0] / img.shape[0])
        x1 = int(x1 * size_reshape[1] / img.shape[1])
        x2 = int(x2 * size_reshape[1] / img.shape[1])
        w = x2 - x1
        h = y2 - y1
        str_result += f" {x1} {y1} {w} {h}"
        start_point = (x1, y1)
        end_point = (x2, y2)
        print(f"Obj: {label}, Start point: {start_point}, End Point: {end_point}")

        color_text = (0, 255, 0)
        color = (0, 0, 255)
        thickness = 1
        font = cv2.FONT_HERSHEY_PLAIN
        font_scale = 0.5
        img_out = cv2.rectangle(img_rs, start_point, end_point, color, thickness)
        img_out = cv2.putText(img_out, str(ind), (x1,y1), font,
                   font_scale, color_text, thickness, 1)

            # text_info_object = f"label: {label} start_point: {start_point} end_point: {end_point}\n"
            # f.write(text_info_object)

        if img_out is not None:
            cv2.imwrite(os.path.join(output_dir,output), img_out)

        # cv2.imshow("yolov3", img_out)
    return str_result + "\n"


def img_blur(filename, v_boxes, v_labels, v_scores, output_dir, size_reshape = (96, 96)):
    img = cv2.imread(filename)
    img_rs = cv2.resize(img, size_reshape, cv2.INTER_AREA)
    blurred_img = cv2.GaussianBlur(img_rs, (35, 35), 0)
    mask = np.zeros((size_reshape[0], size_reshape[1], 3), dtype=np.uint8)
    
    for i in range(len(v_boxes)):
        if not v_boxes:
            x1, y1 = 0, 0
            x2, y2 = 0, 0
        else:
            box = v_boxes[i]
            y1, x1, y2, x2 = box.ymin, box.xmin, box.ymax, box.xmax
            y1 = int(y1 * size_reshape[0] / img.shape[0])
            y2 = int(y2 * size_reshape[0] / img.shape[0])
            x1 = int(x1 * size_reshape[1] / img.shape[1])
            x2 = int(x2 * size_reshape[1] / img.shape[1])

        mask = cv2.rectangle(mask, (x1, y1), (x2, y2), (255, 255, 255), -1)
    out = np.where(mask==np.array([255, 255, 255]), img_rs, blurred_img)
    output = filename.rsplit("/")[1].rsplit(".")[0]+"_blur.jpg"
    # cv2.imwrite("./" + os.path.join(output_dir,output), out)
    # cv2.imshow("img_blur", out)


def split_image(filename, v_boxes, output_dir, number_split = (6, 6), size_reshape = (96, 96)):
    count_img = 0
    label = []
    output = output_dir
    # is_exist = os.path.exists(output + "/img_split/")
    # if not is_exist:
    #     os.makedirs(output + "/img_split/")

    img = cv2.imread(filename)
    img_rs = cv2.resize(img, size_reshape, cv2.INTER_AREA)
    rows = img_rs.shape[1]
    cols = img_rs.shape[0]
    small_box_rows = int(rows / number_split[1])
    small_box_cols = int(cols / number_split[0])
    mask = np.zeros((rows, cols), dtype=np.uint8)

    for i in range(len(v_boxes)):
        if not v_boxes:
            x1,y1 = 0,0
            x2,y2 = 0,0
        else:
            box = v_boxes[i]
            y1, x1, y2, x2 = box.ymin, box.xmin, box.ymax, box.xmax
            y1 = int(y1 * size_reshape[0] / img.shape[0])
            y2 = int(y2 * size_reshape[0] / img.shape[0])
            x1 = int(x1 * size_reshape[1] / img.shape[1])
            x2 = int(x2 * size_reshape[1] / img.shape[1])
        mask = cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

    for row in range(0, img_rs.shape[0], small_box_rows):
        for col in range(0, img_rs.shape[1], small_box_cols):
            matrix_temp = mask[row: row + small_box_rows, col: col + small_box_cols]
            if matrix_temp.max() == 255:
                label.append(1)
            else:
                label.append(0)

    with open(output + '/label_img_split' + '.txt', 'w') as f:
        f.write(str(label))


def draw_split_image(filename, v_boxes, output_dir, number_split = (12, 12), size_reshape = (480, 480)):
    filename_image_split = "label_map_view.png"
    output = output_dir
    # is_exist = os.path.exists(output + "/img_split/")
    # if not is_exist:
    #     os.makedirs(output + "/img_split/")

    img = cv2.imread(filename)
    in_img = img
    img = cv2.resize(img, size_reshape, cv2.INTER_AREA)
    out = img
    rows = img.shape[1]
    cols = img.shape[0]
    small_box_rows = int(rows / number_split[1])
    small_box_cols = int(cols / number_split[0])
    mask = np.zeros((rows, cols), dtype=np.uint8)

    for i in range(len(v_boxes)):
        if not v_boxes:
            x1,y1 = 0,0
            x2,y2 = 0,0
        else:
            box = v_boxes[i]
            y1, x1, y2, x2 = box.ymin, box.xmin, box.ymax, box.xmax
            y1 = int(y1 * size_reshape[0] / in_img.shape[0])
            y2 = int(y2 * size_reshape[0] / in_img.shape[0])
            x1 = int(x1 * size_reshape[1] / in_img.shape[1])
            x2 = int(x2 * size_reshape[1] / in_img.shape[1])

        mask = cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

    for row in range(0, rows, small_box_rows):
        out = cv2.line(out, (0, row + small_box_rows), (rows, row + small_box_rows), (0, 0, 255), 1)
        out = cv2.line(out, (row + small_box_rows, 0), (row + small_box_rows, rows), (0, 0, 255), 1)
        for col in range(0, cols, small_box_cols):
            matrix_temp = mask[row: row + small_box_rows, col: col + small_box_cols]
            if matrix_temp.max() == 255:
                label_small = '1'
            else:
                label_small = '0'

            text_point = (int(col + small_box_cols / 2) - 6, int(row + small_box_rows / 2) + 6)
            out = cv2.putText(out, label_small, text_point, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)

    # cv2.imshow('out', out)
    cv2.imwrite(output + '/' + filename_image_split, out)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, default='',
                        help='path to image file')
    parser.add_argument('--output-dir', type=str, default='outputs/',
                        help='path to the output directory')
    args = parser.parse_args()
    if not os.path.exists(args.output_dir):
        print('==> Creating the {} directory...'.format(args.output_dir))
        os.makedirs(args.output_dir)
    else:
        print('==> Skipping create the {} directory...'.format(args.output_dir))
    return args

########################################################################################################################

# photo_filename = "/home/ttb/Downloads/face-detection-yolov3-keras-main/samples/face_test_yolo.png"
# photo_filename = "/home/ttb/Downloads/face-detection-yolov3-keras-main/samples/Test0.png"
# photo_filename = "/home/ttb/Downloads/face-detection-yolov3-keras-main/samples/face.jpg"
# photo_filename = "/home/ttb/Downloads/face-detection-yolov3-keras-main/samples/group.jpg"
# photo_filename = "/home/ttb/Downloads/face-detection-yolov3-keras-main/samples/test.jpg"

folder_path_list = ["/home/ttb/Downloads/data_yolovX/WIDER_train/WIDER_train/images",
                    "/home/ttb/Downloads/data_yolovX/WIDER_val/WIDER_val/images"]
folder_path_output = "/home/ttb/Downloads/data_yolovX/result"

# define the anchors
anchors = [[116, 90, 156, 198, 373, 326], [30, 61, 62, 45, 59, 119], [10, 13, 16, 30, 33, 23]]

# define the probability threshold for detected objects
class_threshold = 0.7
# define class
labels = ["face"]


def _main():
    # load yolov3 model
    model = load_model('model.h5')
    # Get the arguments
    for folder_path in folder_path_list:
        file_save_label_txt = (folder_path_output + "/" +
                               folder_path.rsplit("/")[-3].rsplit(".")[0]
                               ) + ".txt"
        with open(file_save_label_txt, "w") as fs:
            for path in glob.glob(folder_path + "/*"):
                for photo_filename in glob.glob(path + "/*"):
                    output_dir = (folder_path_output + "/" +
                                  path.rsplit("/")[-3].rsplit(".")[0] + "/" +
                                  photo_filename.rsplit("/")[-2].rsplit(".")[0] + "/" +
                                  photo_filename.rsplit("/")[-1].rsplit(".")[0]) + "/"

                    is_exist = os.path.exists(output_dir)
                    if not is_exist:
                        os.makedirs(output_dir)

                    input_w, input_h = 416, 416
                    str_save_file = ""
                    # load and prepare image
                    image, image_w, image_h = load_image_pixels(photo_filename, (input_w, input_h))

                    yhat = model.predict(image)
                    # summarize the shape of the list of arrays

                    boxes = list()
                    for i in range(len(yhat)):
                        # decode the output of the network
                        boxes += decode_netout(yhat[i][0], anchors[i], class_threshold, input_h, input_w)

                    # correct the sizes of the bounding boxes for the shape of the image
                    correct_yolo_boxes(boxes, image_h, image_w, input_h, input_w)

                    # suppress non-maximal boxes
                    do_nms(boxes, 0.5)  #Discard all boxes with pc less or equal to 0.5

                    # get the details of the detected objects
                    v_boxes, v_labels, v_scores = get_boxes(boxes, labels, class_threshold)

                    # split image
                    split_image(photo_filename, v_boxes, output_dir, number_split=(12, 12), size_reshape = (96, 96))

                    # Draw split image
                    # Image output of function only view so size reshape = (480, 480) or big more
                    draw_split_image(photo_filename, v_boxes, output_dir, number_split=(12, 12), size_reshape = (480, 480))

                    # draw what we found
                    str_save_file = draw_boxes(photo_filename, v_boxes, v_labels, v_scores, output_dir, size_reshape = (96, 96))
                    if str_save_file is not None:
                        fs.write(str_save_file)

                    # blur the rest of image leaving the faces
                    # img_blur(photo_filename, v_boxes, v_labels, v_scores, output_dir, size_reshape = (96, 96))
                    K.clear_session()

if __name__ == "__main__":
    _main()

