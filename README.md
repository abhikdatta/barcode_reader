# barcode_reader
A project for detecting and decoding 1D-barcodes in images

This project is for evaluating 1D barcode detection strategies and not for production.

The current implementation approach is described in [explanation.pdf](https://github.com/abhikdatta/barcode_reader/blob/main/explanation.pdf)

## Dependencies
Dependencies are listed in [environment.yml](https://github.com/abhikdatta/barcode_reader/blob/main/environment.yml). This file can be used to create a compatible conda environment.

## Data
The [Medium Barcode 1D dataset](http://artelab.dista.uninsubria.it/downloads/datasets/barcode/medium_barcode_1d/medium_barcode_1d.html) is used for evaluation. To avoid licensing issues and to keep the repo size small, the dataset is not included in this repo. Before running training and inference scripts, please download the dataset from [here](http://artelab.dista.uninsubria.it/downloads/datasets/barcode/medium_barcode_1d/medium_barcode_1d.zip), and unzip it inside /barcode_reader/data. The scripts expect the data to be available at /barcode_reader/data/BarcodeDatasets.

## Training
To retrain model use the following command: <br>
`python train_classifier.py --recreate_training_data --patch_size 64`

this script uses relative paths, therefore kindly execute it from barcode_reader/src/

## Usage
Before running inference please download the dataset following the steps described in the Dataset section. <br><br>
To run barcode detection use one of the following command: <br>
`python detect_barcode.py --classifier_type cnn --epoch 192  --patch_size 64` <br><br>
`python detect_barcode.py --classifier_type cnn --epoch 194  --patch_size 80`

this script uses relative paths, therefore kindly execute it from barcode_reader/src/
