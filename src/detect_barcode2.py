import os
import numpy as np
import pandas as pd
import cv2
import imutils
from PIL import Image
from pyzbar.pyzbar import decode
import matplotlib.pyplot as plt


def get_files(folder_path, ext=[], return_full_path=False):
    _files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f[-3:] in ext]
    if return_full_path:
        _files = [os.path.join(folder_path, f) for f in _files]
    return _files


def get_msers(gray):
    mser = cv2.MSER_create()
    regions, _ = mser.detectRegions(gray)
    hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
    return regions, hulls


def extract_features(hull):
    
    mask = np.zeros(gray.shape, np.uint8)
    mask = cv2.drawContours(mask, [hull], -1, 255, -1)
    
    mean_intensity = cv2.mean(gray, mask = mask)
    area = cv2.contourArea(hull)

    (x, y), (width, height), angle = cv2.minAreaRect(hull)
    aspect_ratio = max(width, height) / (min(width, height) + 0.01)
    
    features = {'mean_intensity': mean_intensity[0], 
                'area': area, 
                'aspect_ratio': aspect_ratio, 
                'angle': angle}
    
    return features


def classify(feats):
    if feats['mean_intensity'] < 100 and feats['area'] > 1500 and feats['aspect_ratio'] > 10.0:
        return 1
    else:
        return 0


def get_rotated_cropped_image(image, rect):
    
    (x, y), (width, height), angle = rect
    img_copy = image.copy()
    
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    
    Xs = [x[0] for x in box]
    Ys = [y[1] for y in box]
    x1 = min(Xs)
    y1 = min(Ys)
    x2 = max(Xs)
    y2 = max(Ys)

    rotated = False
    if angle < -45:
        angle += 90
        rotated = True

    center = (int((x1+x2)/2), int((y1+y2)/2))
    size = (int(x2-x1), int(y2-y1))

    rotation_matrix = cv2.getRotationMatrix2D((size[0]/2, size[1]/2), angle, 1.0)

    cropped_barcode_reg = cv2.getRectSubPix(img_copy, size, center)
    cropped_barcode_reg = cv2.warpAffine(cropped_barcode_reg, rotation_matrix, size)
    
    cropped_width = width if not rotated else height 
    cropped_height = height if not rotated else width

    cropped_barcode_reg_tight = cv2.getRectSubPix(cropped_barcode_reg, 
                                                  (int(cropped_width), int(cropped_height)), 
                                                  (size[0]/2, size[1]/2))

    return cropped_barcode_reg_tight


def get_barcode_region(hulls):
    
    selected_hulls = []
    for index, hull in enumerate(hulls):    
        features = extract_features(hull)
        if classify(features) == 1:
            selected_hulls.append(hull)

    if len(selected_hulls) == 0:
        return np.zeros((256, 256), np.uint8)
            
    mask = np.zeros(gray.shape, np.uint8)
    mask_all = cv2.drawContours(mask, selected_hulls, -1, 255, -1)
    
    kernel = np.ones((5,5), np.uint8)
    mask_d = cv2.dilate(mask_all, kernel, iterations=10)
    
    contours, hierarchy = cv2.findContours(mask_d, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rect = cv2.minAreaRect(contours[0])
    
    cropped_barcode_reg = get_rotated_cropped_image(gray, rect)
    return cropped_barcode_reg


def show_barcode(cropped_barcode_reg, outpath=None):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(cropped_barcode_reg, cmap='Greys_r', vmin=0, vmax=255)
    
    if outpath is not None:
        plt.savefig(outpath)
        plt.close()
    else:
        plt.show()


def get_ground_truth_barcode(file_name):
    with open(file_name, 'r') as file:
        data = file.read()
    return data



d2 =  get_files("./BarcodeDatasets/Dataset2", ext=["jpg", "JPG"], return_full_path=True)

# for name in ["Dataset1", "Dataset2"]:
for name in ["Dataset1"]:
    d = get_files(os.path.join("BarcodeDatasets", name), ext=["jpg", "JPG"], return_full_path=False)
    for f in d:
        
        image = cv2.imread(os.path.join("BarcodeDatasets", name, f))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        regions, hulls = get_msers(gray)
        cropped_barcode_reg = get_barcode_region(hulls)
        show_barcode(cropped_barcode_reg, 
            outpath=os.path.join("results", name, "barcode_" + f))

        detected_barcodes = decode(cropped_barcode_reg)
        # print(detected_barcodes)

        gt_barcode = get_ground_truth_barcode(os.path.join("BarcodeDatasets", name, f + ".txt"))
        print("GT:", gt_barcode)

        true_detections = 0
        true_barcode_detected = False
        for detected_barcode in detected_barcodes:
            detected_code = detected_barcode.data.decode("utf-8")
            quality = detected_barcode.quality
            print("DETECTED: data={}, quality={}".format(detected_code, quality))
            if detected_code == gt_barcode:
                true_barcode_detected = True
                true_detections += 1
                break

        print("DETECION SUCCESSFUL: ", true_barcode_detected)

    print("DETECION RATE {}: ".format(name), (true_detections*1.0)/(len(f)*1.0))    