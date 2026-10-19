# Face Mask Detection with a CNN

A convolutional neural network that classifies a photo as **masked** or **unmasked**.

## Approach
- Dataset: 1,376 labelled images (690 with mask, 686 without mask)
- Preprocessing: images resized to 300×300 and rescaled to the 0–1 range with Keras `ImageDataGenerator`
- Model: stacked `Conv2D` and `MaxPooling2D` layers followed by dense layers, with a sigmoid output (about 1.7M parameters)
- Training: binary cross-entropy with RMSprop, and a callback that stops training once accuracy passes 95%

## Result
Reached 96.7% training accuracy. The notebook tests the model on two sample photos (one masked, one unmasked) and classifies both correctly. A separate validation or test split has not been evaluated yet.

## Files
- `Facemask_detection.ipynb`: the full notebook
- `Facemask_detection - Jupyter Notebook.pdf`: a PDF copy of the notebook

## Tech
Python, TensorFlow, Keras, OpenCV, NumPy, Matplotlib
