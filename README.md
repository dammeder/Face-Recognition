# Raspberry Pi Facial Recognition System

A facial recognition system built on a **Raspberry Pi 4** with a **Pi Camera**.  
Currently, it focuses on **training and running a facial recognition model**.  
Future plans include integrating it with a smart door-lock system.

---

## 🔹 Project Overview

- Captures images and video using the Pi Camera  
- Trains facial encodings from a local dataset  
- Performs real-time facial recognition  

---

## 🔹 Current Features

- Face image capture for dataset creation  
- Facial encoding training  
- Real-time face detection and recognition  
- Name matching using pre-trained encodings
- Adjustable frame processing rate — the system can skip frames for faster performance or slower CPU usage
- Real-time FPS display for monitoring system performance
  
---

## 🔹 Hardware Used

- Raspberry Pi 4  
- Raspberry Pi Camera Module

---

## 🔹 Software & Libraries

- Python 3.13  
- OpenCV (`opencv-python`)  
- face_recognition  
- NumPy  
- imutils

---

## 🔹 Dataset Structure

Organize your images by person name:

```text
dataset/
├── PERSON_NAME/
│ ├── PERSON_NAME_20240101_120000.jpg
│ └── PERSON_NAME_20240101_120001.jpg
├── ANOTHER_PERSON/
│ ├── ANOTHER_PERSON_20240101_120000.jpg
│ └── ANOTHER_PERSON_20240101_120001.jpg
```


> Each folder represents one individual used to train the model.  


---

## 🔹 How to Run

1. **Install dependencies**

```bash
# Install dependencies 
pip install -r requirements.txt
# Capture face images 
python image_capture.py

# Train model --- Press SPACE to take picture ( 20-30 per person works fine, take pictures from different angles & different lighting ), 
python model_training.py

# Press q to exit the video window.

# Run model
# In facial_recognition.py, you can change how often faces are processed:
# process_every_n_frames = 5
# Increase the number to process less often (less CPU), decrease to process more often (faster recognition) -- I found that anything larger than 10 wont be able to keep up with normal movement
# With none of the code changed you get around 6-10 FPS with face in frame.
# cv_scale can also be changed based on the distance you would like the model to recognize faces from. cv_scale trades recognition distance for speed — higher is faster, lower sees farther.
python facial_recognition.py

# Press q to exit the video window.
