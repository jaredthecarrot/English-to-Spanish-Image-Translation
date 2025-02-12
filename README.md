# English-to-Spanish Translation Model

This project is a language translation model that translates text from English to Spanish using a combination of Image Processing and Natural Language Processing. The model captures text from images and translates it into the desired language.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
  - [Language Requirements](#language-requirements)
  - [Hardware Requirements](#hardware-requirements)
  - [Needed Libraries](#needed-libraries)
  - [Programs Needed](#programs-needed)
  - [Pre-requisites](#pre-requisites)
- [Notes](#notes)
- [Known Issues](#known-issues)
- [Steps](#steps)

## Overview

This project uses a combination of **Image Processing** and **Natural Language Processing** to extract text from photos and translate it into Spanish. The main components include:

- **Image processing** for extracting text from images using PyTesseract.
- **Natural Language Processing** to train a model for translation.

## Installation

### Language Requirements
- Python version 3.9 or higher
- Juypter Notebook

### Hardware Requirements:
- NVIDIA GPU with CUDA support
- Windows 10 or higher

### Needed Libraries:
- **OpenCV**
- **PyTesseract**
- **Numpy**
- **TensorFlow**
- **Keras**

**Note:** OpenCV, PyTesseract, and TensorFlow will require additional setup, and links for downloading will be provided below.

### Programs Needed:

- **OpenCV:** [Installation link](https://opencv.org/releases/)
- **PyTesseract:** [Installation link](https://github.com/h/pytesseract)
- **TensorFlow:** [Installation link](https://www.tensorflow.org/install)

### Pre-requisites:
- Update `pip` using the following command:
  ```bash
  pip install --upgrade pip

## Notes
- Expect some delay between image processing and text extraction.
- Handwritten text is highly likely to not be recognized by the image extraction program.

## Installation Instructions

### OpenCV:
1. Download and install OpenCV from [here](https://opencv.org/releases/).
2. Ensure that the pathing is correctly set in your system (a restart may be required).
3. Install OpenCV via pip:
   ```bash
   pip install opencv-python
4. Verify the installation
   '''bash
   import cv2

### PyTesseract:
1. Download and extract PyTesseract from [GitHub](https://github.com/h/pytesseract).
2. Ensure that the pathing is correctly set in your system (a restart may be required).
3. Install PyTesseract via pip:
   ```bash
   pip install pytesseract
4. Verify the installation
   '''bash
   import pytesseract

### TensorFlow:
1. Ensure that your hardware and software meet TensorFlow's requirements.
2. Install TensorFlow via pip:
   ```bash
   pip install tensorflow
3. Verift the installation
   '''bash
   import tensorflow

### Numpy:
1. Install Numpy via pip:
   ```bash
   pip install numpy
   
### Keras:
1. Install Keras via pip:
   ```bash
   pip install keras

### Known Issues:
- **PyTesseract** requires proper pathing. Ensure the PyTesseract files are correctly installed and pathed.
- The translation model cannot currently save the trained Keras model for future translations. It needs to be retrained (~8 hours) for experiments.
- The camera capture functionality may not respond correctly or capture from the connected camera.


## Steps
  Clone the repository for the translation model:
   ```bash
git clone https://github.com/yourusername/your-repository.git
cd your-repository
The project includes three repositories for different image input methods:
- **Pytesseract1 Repository**: Uses files saved on the computer (manual input required in the code).
- **Pytesseract2 Repository**: Uses image URLs for text extraction.
- **Pytesseract3 Repository**: Uses a connected camera for image capture (this is experimental and may not work correctly).

Once you have cloned the repository, open `transformer.ipynb` to start using the model.


