# barcode_reader
A sample project for detecting and decoding 1D-barcodes in images

This project is for evaluating 1D barcode detection strategies and not for production.

To learn more about the current processing logic check src/strong_barcode_reader.ipynb

Dependencies are listed in environment.yml. This file can also be used to create a compatible conda environment.

To retrain model use one the following command: <br>
`python train_classifier.py --recreate_training_data --patch_size 64`

this script uses relative paths, therefore kindly execute it from barcode_reader/src/

To run barcode detection use one of the following command: <br>
`python detect_barcode.py --classifier_type cnn --epoch 197  --patch_size 64` <br>
`python detect_barcode.py --classifier_type cnn --epoch 194  --patch_size 80`

this script uses relative paths, therefore kindly execute it from barcode_reader/src/
