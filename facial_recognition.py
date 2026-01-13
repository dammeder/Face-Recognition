import face_recognition
import cv2
import numpy as np
from picamera2 import Picamera2
import time
import pickle

# scale the live feed down - less pixels - faster detection
# find faces - compare - 
# scale up the feed so face boxes are accurate 


with open("encodings.pickle", "rb") as f:
    data = pickle.loads(f.read())
    
known_face_encoding = data["encondings"]
known_face_names = data["name"]

picam = Picamera2()
picam.configure(picam.create_preview_configuration(main={"format": "XRGB8888", "size": (640, 480)}))
picam.start()

cv_scaler = 3 

face_locations = []
face_encodings = []
face_names = []
frame_count = 0
start_time = time.time()
fps = 0


process_every_n_frames = 5
frame_number = 0

def process_frame(frame):
    global face_locations, face_encodings, face_names, process_every_n_frames, frame_number
    
    frame_number += 1
    
    resized_frame = cv2.resize(frame, (0,0), fx = (1/cv_scaler) ,fy = (1/cv_scaler))
    
    rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    
        
    if frame_number % process_every_n_frames == 0:
        
        face_locations = face_recognition.face_locations(rgb_frame, model = "hog")
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations, model = "large")
    
   # face_names = []
    
        tolerance = 0.40
        face_names = []
    
        for face_encoding in face_encodings:
            
            #name = "UNKOWN"        
            
            matches = face_recognition.compare_faces(known_face_encoding, face_encoding)
            face_distances = face_recognition.face_distance(known_face_encoding, face_encoding)
            
            
            best_match_index = np.argmin(face_distances)
            best_distance = face_distances[best_match_index]
            
            #tolerance = 0.30
            
            
            
            print(best_distance)
            
            if best_distance < tolerance:
                name = known_face_names[best_match_index]
            else:
                name = "UNKOWN"
                
            face_names.append(name)
        
    return frame

def draw_results(frame):
    
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= cv_scaler
        right *= cv_scaler
        bottom *= cv_scaler
        left *= cv_scaler
    
    
        cv2.rectangle(frame, (left, top), (right, bottom), (244, 42, 3), 5)
        
        cv2.rectangle(frame, (left - 3, top -35), (right + 3, top), (244, 42, 3), cv2.FILLED)
        
        font = cv2.FONT_HERSHEY_DUPLEX
        
        cv2.putText(frame, name, (left + 6, top -6), font, 1.0, (255, 255, 255), 1)
    
        
    return frame 
                
        
def calc_fps():
    
    global frame_count, start_time, fps
    
    frame_count +=1
    elapsed_time = time.time() - start_time
    
    if elapsed_time > 1:
        fps = frame_count / elapsed_time
        frame_count = 0
        start_time = time.time()
        
    return fps

while True:
    frame = picam.capture_array()
    
    
    processed_frame = process_frame(frame)
    
    display_frame = draw_results(processed_frame)
    
    current_fps = calc_fps()
    #cv2.putText(frame, name, (left + 6, top -6), font, 1.0, (255, 255, 255), 1)
 #  ret, frame = picam.read()
    
    cv2.putText(display_frame, f"FPS: {current_fps:.1f}", (display_frame.shape[1] -150, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
    
    cv2.imshow("Video", display_frame)
    
    if cv2.waitKey(1) == ord('q'):
        break
    
    
    
cv2.destroyAllWindows()
picam.stop()
    
    
    
    
    
    
    
    
    
    
    
    