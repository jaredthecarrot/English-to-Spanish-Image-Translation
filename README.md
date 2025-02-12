## Overview
This project is a language translation model from English-to-Spanish. 
It uses a combination of Image Processing and Natrual Language Processing to capture text from photos and translate them into a different language.
---------------------------------------------------------------------------------------------------------------------------------------------------
## Installation
#Language
  -Python
    -Version 3.9 or higher
#Hardware Requirements
  -NVIDIA GPU with CUDA
  -Windows 10 or higher
---------------------------------------------------------------------------------------------------------------------------------------------------  
#Needed Libraries
  -OpenCV
  -PyTesseract
  -Numpy
  -tensorflow
  -keras
  
OpenCV, PyTesseract, and Tensorflow will need additional downloaded content. Links attached below.

#Programs Needed
OpenCV: https://opencv.org/releases/
PyTesseract: https://github.com/h/pytesseract
Tensorflow: https://www.tensorflow.org/install

##Pre-requisties
-Check 'pip' for updates using command -pip install --upgrade pip
#OpenCV
-Download and Install OpenCV
-Ensure Pathing is established in system (May need system restart)
-Check version of opencv-python
-use command -pip install opencv-python in python terminal
-use command -import opencv-python as cv2
#PyTesseract
-Download and Extract PyTesseract files
-Ensure Pathing is established in system (May need system restart)
-Check version of Pytesseract
-use command -pip install pytesseract in python terminal
-use command import pytesseract
#Tensorflow
-Ensure hardware and software requirements
-use -pip install tensorflow in python terminal
-use command import tensorflow
#Numpy
-use command -pip install numpy
-import numpy
#keras
-use command -pip install keras
-import keras

##Steps
-Clone reposititory for transformer.ipynd 
  -Repositories for Pytesseract1 uses files saved from computer (Will need manual input in code to use)
  -Repository for Pytesseract2 uses image urls
  -Reposititory for Pytesseract3 uses camera connected to system for image capture (Experimental and not fully functional)

## NOTES
-Current program for transformer model includes language training and translation of select target sentences
-Program was created in a very small time frame
#Known Issues
-Pytesseract models will require file pathing, ensure pytesseract files are installed and correctly pathed
-Translation model unable to save keras model for future translations, current model requires retraining for experiments (~8 hours)
-Camera capture function may not repsond correctly or capture connected camera
-Expect some delay between image processing and text extraction
-Handwritten text highly likely to not be recognized by image extraction program
