import os
import numpy as np
import pandas as pd
import cv2
import imutils
from PIL import Image
import skimage.measure
from pyzbar.pyzbar import decode
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tensorflow as tf
import argparse


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


def get_barcode_region_cnn(model, gray, tile_size=(64,64), save_detection_results=False, file_name=None):

    _gray = gray.copy()
    _gray = _gray.astype(np.float32)
    _gray /= 255.0

    H = tile_size[0]
    W = tile_size[1]
    slices = dict([((x,y), np.expand_dims(_gray[y:y+H,x:x+W], axis=2)) for y in range(0,_gray.shape[0],H) for x in range(0,_gray.shape[1],W)])
    
    positive_coords = []
    negative_coords = []
    mask = np.zeros(gray.shape, np.uint8)

    for index, (x,y) in enumerate(slices):
        sz = slices[(x,y)].shape
        if sz[0] == H and sz[1] == W:
            if classify_cnn(model, slices[(x,y)]) == 1:
                mask[y:y+H,x:x+W] = 1
                positive_coords.append((x,y))
            else:
                negative_coords.append((x,y))

    if len(positive_coords) == 0:
        return np.zeros((256, 256), np.uint8)

    kernel = np.ones((5,5), np.uint8)
    mask_d = cv2.dilate(mask, kernel, iterations=10)

    if save_detection_results:
        save_detections(gray, positive_coords, negative_coords, tile_size, os.path.join("../results/debug/", "detection_" + file_name))
        plot(mask, os.path.join("../results/debug/", "detection_mask_" + file_name))
        plot(mask_d, os.path.join("../results/debug/", "detection_mask_dilated" + file_name))

    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    # closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    cropped_barcode_regions = []
    # N, labels = cv2.connectedComponents(mask, 4, cv2.CV_32S)

    if save_detection_results:
        selected_contours = []

    labeled_image, count = skimage.measure.label(mask_d, connectivity=2, return_num=True)
    for label in range(1, count):
        mask_i = np.zeros((labeled_image.shape[0], labeled_image.shape[1]), dtype=np.uint8)
        mask_i[labeled_image == label] = 1
        area = np.sum(mask_i)
        if area > 2500:
            contours, hierarchy = cv2.findContours(mask_i, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            rect = cv2.minAreaRect(contours[0])
            cropped_barcode_reg = get_rotated_cropped_image(gray, rect)
            cropped_barcode_regions.append(cropped_barcode_reg)
            if save_detection_results:
                selected_contours.append(contours)

    if save_detection_results:
        save_selected_detections(selected_contours, mask_d, os.path.join("../results/debug/", "min_area_rect_" + file_name))

    return cropped_barcode_regions


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


def load_model(model_dir, epoch):
    model = tf.keras.models.load_model(os.path.join(model_dir, "{0:02d}.h5".format(epoch)))
    return model


def classify_cnn(model, image):
    p = model(np.array([image]), training=False)
    return np.argmax(p[0])


def save_detections(gray, positive_coords, negative_coords, tile_size, outpath):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(gray, cmap='Greys_r', vmin=0, vmax=255)

    for (x,y) in positive_coords:
        rect = patches.Rectangle((x,y), tile_size[0], tile_size[1], linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    for (x,y) in negative_coords:
        rect = patches.Rectangle((x,y), tile_size[0], tile_size[1], linewidth=1, edgecolor='g', facecolor='none')
        ax.add_patch(rect)
    
    plt.savefig(outpath)
    plt.close()


def save_selected_detections(selected_contours, mask_d, outpath):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    canvas = np.ones((mask_d.shape[0], mask_d.shape[1], 3), dtype=np.int8)*150
    for _contours in selected_contours:
        cv2.drawContours(canvas, _contours, -1, (0, 100, 0), -1)
        rect = cv2.minAreaRect(_contours[0])
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(canvas,[box],0,(255,0,0),3)
    
    plt.imshow(canvas)
    plt.savefig(outpath)
    plt.close()


def plot(mask, outpath):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(mask, cmap='Greys_r', vmin=0, vmax=1)
    plt.savefig(outpath)
    plt.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Strong barcode detection")
    parser.add_argument("--classifier_type", help="The classifier type to use", type=str, choices=["cnn", "features"], default="cnn")
    parser.add_argument("--epoch", help="The model to use", type=int, default=1)
    parser.add_argument("--patch_size", help="Patch size to use during training", type=int, default=64)
    parser.add_argument("--debug", help="Run in debug mode", action="store_true")
    args = parser.parse_args()


    if args.classifier_type == 'cnn':
        _model = load_model("../models/barcode_detector", args.epoch)
        _tile_size = (args.patch_size, args.patch_size)
    else:
        _model = None

    if args.debug:
        if not os.path.isdir("../results/debug"):
            os.makedirs("../results/debug")

    for name in ["Dataset1", "Dataset2"]:
    # for name in ["Dataset1"]:
        processed_cnt = 0
        true_detections = 0
        d = get_files(os.path.join("../data/BarcodeDatasets", name), ext=["jpg", "JPG"], return_full_path=False)
        for f in d:
            
            image = cv2.imread(os.path.join("../data/BarcodeDatasets", name, f))
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            if args.classifier_type == 'features':
                regions, hulls = get_msers(gray)
                cropped_barcode_reg = get_barcode_region(hulls)
                cropped_barcode_regions = [cropped_barcode_reg]
            
            elif args.classifier_type == 'cnn':
                cropped_barcode_regions = get_barcode_region_cnn(_model, gray, tile_size=_tile_size, save_detection_results=args.debug, file_name=f)

            else:
                raise ValueError("Unknown classifier type")

            detected_barcodes = []
            for instance_index, cropped_barcode_reg in enumerate(cropped_barcode_regions):
                if args.debug:
                    show_barcode(cropped_barcode_reg, 
                                 outpath=os.path.join("../results/debug", "barcode_{}_".format(instance_index) + f))

                _detected_barcodes = decode(cropped_barcode_reg)
                detected_barcodes.extend(_detected_barcodes)
            # print(detected_barcodes)

            gt_barcode = get_ground_truth_barcode(os.path.join("../data/BarcodeDatasets", name, f + ".txt"))
            print("GT:", gt_barcode)

            true_barcode_detected = False
            for detected_barcode in detected_barcodes:
                detected_code = detected_barcode.data.decode("utf-8")
                quality = detected_barcode.quality
                print("DETECTED: data={}, quality={}".format(detected_code, quality))
                if detected_code == gt_barcode:
                    true_barcode_detected = True

            if true_barcode_detected:
                true_detections += 1

            processed_cnt += 1
            print("{}/{}: Detection Successful = {}, Current Detection Rate = {}".format(name, f, true_barcode_detected, (true_detections*1.0)/(processed_cnt*1.0)))
            