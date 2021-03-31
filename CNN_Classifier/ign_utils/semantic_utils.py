import cv2
import numpy as np

def encode_labeldict(labelmap_dict):
    new_dict= {}
    label_list = sorted(list(labelmap_dict.keys()))
    for i in range (len(label_list)):
        templist = labelmap_dict[label_list[i]]
        templist = list(np.array(templist).astype(np.int64))
        new_dict[str(i+1)] = templist
    new_dict[str(0)] = [0, 0, 0]
    return new_dict

def get_contours(mask):
    contours = None
    #print('Mask Shape: ', mask.shape)
    im_gray = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    if cv2.countNonZero(im_gray) > 0:
        # display_image(im_gray, name='mask', waitkey=0)
        ret, open_mask = cv2.threshold(im_gray, 0, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(
            open_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def get_min_rect(cnt):
    """
    returns four corner points fitting given given contour
    """
    if cnt is None:
        return None
    return make_fit_rect(cnt)
    approx_rect = None
    epsilon = 0.0001
    # print('iteration starts')
    cnt = cv2.convexHull(cnt, returnPoints=True)  # also to reduce points
    while 1:
        epsilon += .1
        # print('epsilon',epsilon)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        # print(len(approx))
        if len(approx) < 4:
            break
        elif len(approx) == 4:
            approx_rect = approx
            break
        else:
            pass
    # print(len(approx))
    # print('rect points:',approx_rect)
    return approx_rect

def make_fit_rect(box):
    rect = cv2.minAreaRect(np.array(box))
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    return box

def get_iou(y_true=None, y_pred=None, n_classes=None, colormapdict=None):

    for class_key, color_value in colormapdict.items():
        y_true[(y_true==color_value).all(axis=2)] = class_key
        y_pred[(y_pred==color_value).all(axis=2)] = class_key

    class_wise = np.zeros(n_classes)
    for cl in range(n_classes):
        intersection = np.sum((y_true == cl)*(y_pred == cl))
        union = np.sum(np.maximum((y_true == cl), (y_pred == cl)))
        iou = float(intersection)/(union)
        class_wise[cl] = iou
    return class_wise


