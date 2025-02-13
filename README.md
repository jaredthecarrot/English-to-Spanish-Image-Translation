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
- [Examples](#examples)

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
- Must run program without Debugging. If ran while debugging program freezes at text extraction until debugging ends.
- Tesseract Model does not some Spanish characters

## Steps
  Clone the repository for the translation model:
  
git clone https://github.com/yourusername/your-repository.git
cd your-repository
The project includes three repositories for different image input methods:
- **Pytesseract1 Repository**: Uses files saved on the computer (manual input required in the code).
- **Pytesseract2 Repository**: Uses image URLs for text extraction.
- **Pytesseract3 Repository**: Uses a connected camera for image capture (this is experimental and may not work correctly).

Once you have cloned the repository, open `transformer.ipynb` to start using the model.

# Examples
## PRINTED TEST
### Input
![testocr](https://github.com/user-attachments/assets/ede32670-8531-48d4-9c28-e7a19bf0bdb8)
### Output
<img width="858" alt="image" src="https://github.com/user-attachments/assets/0d6c2ba6-4390-4433-9ccb-2bfa31a06429" />

## HAND WRITTEN TEXT
### Input
![image](https://github.com/user-attachments/assets/6a696b6d-9ae7-45f5-800f-1af2475fbfa2)
### OutPut
<img width="859" alt="image" src="https://github.com/user-attachments/assets/90d69a72-f2ab-41b8-b953-fcbc60a29256" />

## LOW QUALITY IMAGE
### Input
![image](https://github.com/user-attachments/assets/2d96f99d-8d20-434f-b256-1d932a196d75)
### Output
<img width="862" alt="image" src="https://github.com/user-attachments/assets/fa0e9769-9436-47f1-ad03-fd5e8bea5f25" />

## MULTI-LINUGAL
### Input
![image](https://github.com/user-attachments/assets/458749f9-ab29-432a-b04b-70c77d672302)
### Output
<img width="854" alt="image" src="https://github.com/user-attachments/assets/76cbce1e-d78e-4881-99c7-665de4c09d66" />
