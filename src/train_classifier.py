import os
import sys
import numpy as np
import pandas as pd
from lxml import objectify
from sklearn.model_selection import train_test_split
import urllib
import shutil
from PIL import Image
import argparse

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten

sys.path.insert(0, "./data_augmentation")

from data_aug.data_aug import *
from data_aug.bbox_util import *

from detect_barcode import get_files



def create_csv_annotation(dataset_src, labels_dir):

    d1 =  get_files(os.path.join(dataset_src, "Dataset1"), ext=["jpg", "JPG"], return_full_path=False)
    d2 =  get_files(os.path.join(dataset_src, "Dataset2"), ext=["jpg", "JPG"], return_full_path=False)
    label_files =  get_files(labels_dir, ext=["xml"], return_full_path=True)

    annotations = {'image_name': [], 'x1': [], 'y1': [], 'x2': [], 'y2': [], 'class': [], 'dataset': []}
    for _label_file in label_files:

        with open(_label_file, 'r') as file:
            anno = objectify.parse(_label_file).getroot()
            
        file_name = anno["filename"][0].text
        print(file_name)

        if file_name in d1:
            if file_name in d2:
                raise ValueError("Filename match not found in both datasets: " + file_name)
            else:
                annotations['dataset'].append("Dataset1")
        elif file_name in d2:
            annotations['dataset'].append("Dataset2")
        else:
            raise ValueError("Filename match not found in either dataset: " + file_name)

        annotations['image_name'].append(file_name)
        annotations['x1'].append(anno["object"][0]["bndbox"][0]["xmin"][0])
        annotations['y1'].append(anno["object"][0]["bndbox"][0]["ymin"][0])
        annotations['x2'].append(anno["object"][0]["bndbox"][0]["xmax"][0])
        annotations['y2'].append(anno["object"][0]["bndbox"][0]["ymax"][0])
        annotations['class'].append('barcode')
        
    return annotations


def get_test_val_split(annotations, dataset_src, dataset_dest, seed=1783):

    df = pd.DataFrame.from_dict(annotations, orient='columns')
    
    df_train, df_val = train_test_split(df, test_size=0.33, random_state=seed, stratify=df['dataset'])
    df_train.drop('dataset', axis=1).to_csv(os.path.join(dataset_dest, "annotations_train.csv"), header=False, index=False)
    
    df_val['image_name'] = dataset_src + "/" + df_val['dataset'] + "/" + df_val['image_name']
    df_val.drop('dataset', axis=1).to_csv(os.path.join(dataset_dest, "annotations_val.csv"), header=False, index=False)

    return df_train, df_val


def augment(df, dataset_src, dataset_dest, num_augments_per_sample=5):

    annotations = {'image_name': [], 'x1': [], 'y1': [], 'x2': [], 'y2': [], 'class': []}

    for index, row in df.iterrows():

        # copy original image
        src_filename = os.path.join(dataset_src, row['dataset'], row['image_name'])
        dest_filename = os.path.join(dataset_dest, row['image_name'])
        shutil.copyfile(src_filename, dest_filename)

        # update annotations for original image
        annotations['image_name'].append(row['image_name'])
        annotations['x1'].append(row['x1'])
        annotations['y1'].append(row['y1'])
        annotations['x2'].append(row['x2'])
        annotations['y2'].append(row['y2'])
        annotations['class'].append(row['class'])

        # read image and convert from bgr to rgb
        img = cv2.imread(dest_filename)[:,:,::-1]

        for i in range(num_augments_per_sample):
            bboxes = np.array([[row['x1'], row['y1'], row['x2'], row['y2'], 0]])
            seq = Sequence([RandomHSV(40, 40, 30), RandomHorizontalFlip(), RandomScale(), RandomTranslate(), RandomRotate(10), RandomShear()])
            img_, bboxes_ = seq(img.copy(), bboxes)

            # save augmented image
            parts = row['image_name'].split(".")
            aug_filename = "{}_aug{}.{}".format(parts[0], i, parts[1])
            cv2.imwrite(os.path.join(dataset_dest, aug_filename), img_)

            # add annotations for augmented image
            for box_ in bboxes_:
                annotations['image_name'].append(aug_filename)
                annotations['x1'].append(box_[0])
                annotations['y1'].append(box_[1])
                annotations['x2'].append(box_[2])
                annotations['y2'].append(box_[3])
                annotations['class'].append(row['class'])

    # save df
    augmented_df = pd.DataFrame.from_dict(annotations, orient="columns")
    augmented_df.to_csv(os.path.join(dataset_dest, "annotations_train_augmented.csv"), header=False, index=False)
    return augmented_df


def proportion_of_B_in_A(boxA, boxB):

    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxBArea)
    # return the intersection over union value
    return iou


class AnnotationGenerator(tf.keras.utils.Sequence):

    def __init__(self, annotations_df, images_per_batch, epochs, samples_per_image, base_data_dir=None, shuffle=True, tile_size=(64,64)):
        self._annotations_df = annotations_df
        self._images_per_batch = images_per_batch
        self._samples_per_image = samples_per_image
        self._base_data_dir = base_data_dir
        self._shuffle = shuffle
        self._tile_size = tile_size
        self._batch_size = samples_per_image*images_per_batch*2
        self._num_epochs = epochs
        self._current_epoch = -1
        self.on_epoch_end()

    @property
    def batch_size(self):
        return self._batch_size

    def read_image(self, file_path, bboxes=None):

        # read image
        img = np.array(Image.open(file_path))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype('float32')
        gray /= 255.0

        # slice image into tiles
        H = self._tile_size[0]
        W = self._tile_size[1]
        slices = dict([((x,y), np.expand_dims(gray[y:y+H,x:x+W], axis=2)) for y in range(0,gray.shape[0],H) for x in range(0,gray.shape[1],W)])

        positives = []
        negatives = []
        positive_coords = []
        negative_coords = []
        if bboxes is not None:
            # for each slice assign label
            for (x,y) in slices:
                sz = slices[(x,y)].shape
                if sz[0] == H and sz[1] == W:
                    # assumin only 1 bbox
                    inter_prop = proportion_of_B_in_A(bboxes[0], [x, y, x+W, y+H])
                    if inter_prop > 0.5:
                        positives.append(slices[(x,y)])
                        positive_coords.append((x,y))
                    else:
                        negatives.append(slices[(x,y)])
                        negative_coords.append((x,y))

        return positives, negatives, positive_coords, negative_coords

    def __len__(self):
        return int(np.floor(len(self._annotations_df)/self._images_per_batch))

    def __getitem__(self, index):
        """Generate one batch of data
        """
        current_indices = np.arange(index*self._images_per_batch, (index+1)*self._images_per_batch, 1)
        samples = []
        labels = []
        for index in current_indices:

            row = self._annotations_df.iloc[index]
            if self._base_data_dir is None:
                fname = row['image_name']
            else:
                fname = os.path.join(self._base_data_dir, row['image_name'])

            # read data
            positives, negatives, _, _ = self.read_image(fname, [[row['x1'], row['y1'], row['x2'], row['y2']]])

            # sample
            indices = np.random.randint(0, high=len(positives), size=self._samples_per_image)
            samples.extend([positives[i] for i in indices])
            labels.extend([1]*len(indices))

            indices = np.random.randint(0, high=len(negatives), size=self._samples_per_image)
            samples.extend([negatives[i] for i in indices])
            labels.extend([0]*len(indices))

        # shuffle
        samples_labels = list(zip(samples, labels))
        random.shuffle(samples_labels)
        samples, labels = zip(*samples_labels)

        samples = np.asarray(samples).astype('float32')

        one_hot_labels = np.zeros((len(labels), 2))
        for i, label_ in enumerate(labels):
            one_hot_labels[i][label_] = 1

        return (samples, one_hot_labels)
    
    def on_epoch_end(self):
        if self._shuffle:
            self._annotations_df = self._annotations_df.sample(frac=1).reset_index(drop=True)
        self._current_epoch += 1



class ModelSaveCallback(tf.keras.callbacks.Callback):

    def __init__(self, model_dir):
        self._model_dir = model_dir

    def on_epoch_end(self, epoch, logs=None):
        self.model.save(os.path.join(self._model_dir, "{0:02d}.h5".format(epoch)))


def create_model(img_shape, lr=0.01):

    strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")
    with strategy.scope():
        input = keras.Input(shape=img_shape)
        x = Conv2D(16, (3,3), padding="same", activation="relu", name="conv_0")(input)
        x = MaxPooling2D(pool_size=(2,2), name="pool_0")(x)
        x = Conv2D(32, (3,3), padding="same", activation="relu", name="conv_1")(x)
        x = MaxPooling2D(pool_size=(2,2), name="pool_1")(x)
        x = Conv2D(32, (3,3), padding="same", activation="relu", name="conv_2")(x)
        x = MaxPooling2D(pool_size=(2,2), name="pool_2")(x)
        x = Flatten()(x)
        x = Dense(100, activation="relu")(x)
        x = Dropout(0.5)(x)
        output = Dense(2, activation="softmax")(x)

        model = keras.Model(input, output, name="barcode")
        model.compile(optimizer=keras.optimizers.Adagrad(), loss="binary_crossentropy", metrics=["accuracy", "binary_accuracy"])

    model.summary()
    return model


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="barcode detection")
    parser.add_argument("--recreate_training_data", help="Whether to re-create training cache", action="store_true")
    parser.add_argument("--patch_size", help="Patch size to use during training", type=int, default=64)
    args = parser.parse_args()

    seed = 1783
    dataset_root = "../data/BarcodeDatasets"
    training_dataset = "../data/training_data"
    model_dir = "../models/barcode_detector"

    if args.recreate_training_data:
        annotations = create_csv_annotation(dataset_root, "../data/labels")
        df_train, df_val = get_test_val_split(annotations, dataset_root, training_dataset, seed=seed)
        augmented_df = augment(df_train, dataset_root, training_dataset, num_augments_per_sample=9)
    else:
        augmented_df = pd.read_csv(os.path.join(training_dataset, "annotations_train_augmented.csv"),
                                   names=['image_name', 'x1', 'y1', 'x2', 'y2', 'class'])

        df_val = pd.read_csv(os.path.join(training_dataset, "annotations_val.csv"),
                             names=['image_name', 'x1', 'y1', 'x2', 'y2', 'class'])

    samples_per_image_ = 5
    images_per_batch_ = 4
    epochs_ = 200
    tile_size_ = (args.patch_size, args.patch_size)

    train_generator = AnnotationGenerator(augmented_df, 
                                          images_per_batch_, 
                                          epochs_, 
                                          samples_per_image_, 
                                          base_data_dir=training_dataset, 
                                          tile_size=tile_size_)

    val_generator = AnnotationGenerator(df_val, 
                                        images_per_batch_, 
                                        epochs_, 
                                        samples_per_image_, 
                                        base_data_dir=None, 
                                        tile_size=tile_size_)


    model = create_model([tile_size_[0], tile_size_[1], 1])

    saver = ModelSaveCallback(model_dir)
    history_logger=tf.keras.callbacks.CSVLogger("../results/training_log.csv", separator=",", append=True)

    _steps_per_epoch = int(np.floor(len(augmented_df)*1.0/images_per_batch_))

    model.fit(x=train_generator, 
              validation_data=val_generator, 
              steps_per_epoch=_steps_per_epoch, 
              batch_size=train_generator.batch_size, 
              epochs=epochs_, 
              callbacks=[saver, history_logger])

