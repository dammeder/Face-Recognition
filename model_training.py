import cv2
import numpy as np
import os
import face_recognition
from imutils import paths
import pickle

image_paths = list(paths.list_images("dataset"))
known_encdodings = []
known_names = []
image_count = len(image_paths)

# print(image_paths)

for (i, image_paths) in enumerate(image_paths):
    print(f"processing image number {i + 1}/{image_count}")
    
    ## output == dataset/MEDER/MEDER_20260102_045155.jpg
    ##            0       1                2          3
    
    name = image_paths.split(os.path.sep)[1]
    
    #print(image_paths)
    
    img = cv2.imread(image_paths) # get the array value of each image 
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
    
    
    boxes = face_recognition.face_locations(rgb, model = "hog") # extract face from each img 
    encodings = face_recognition.face_encodings(rgb, boxes) # numeric fingerprint for each face 
    
    for encoding in encodings:
        known_encdodings.append(encoding)
        known_names.append(name)


#print(known_encdodings,known_names)
        
data = {"encondings":known_encdodings, "name":known_names}
    
with open("encodings.pickle", "wb") as f:
    f.write(pickle.dumps(data)) # pickle isntead of json bc enconding are NumPy arrays -- json cant represnt them prooperly
    
print("TRAINING COMPLETE, ENCONDINGS SAVED TO encondings.pickle")
    
    
    