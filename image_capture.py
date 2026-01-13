import cv2
import os
from datetime import datetime as dt
from picamera2 import Picamera2
import time


# --------> CAPITAL NAMES ONLY <--------

person_name = "ERJAN"

def create_folder(subject):
    dataset_folder = "dataset"
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder) #make a folder to store pictures
        
    person_folder = os.path.join(dataset_folder, subject)
    if not os.path.exists(person_folder):
        os.makedirs(person_folder) #make a subfolder of each persons photos
    return person_folder
    
def capture_photos(subject):
    folder = create_folder(subject)
    
    picam = Picamera2()
    # start the cam ; XRGB8888 = how it should be stored in memory ;
    # since PI CPU can only process 4 or 8 bits at time just RGB will be awkward so the X is used as "padding"  
    # so RGB is for all the colors and X --- XRGB plus each letters bits - XRGB8888
    picam.configure(picam.create_preview_configuration(main={"format": "XRGB8888", "size": (640, 480)}))
    picam.start()
    
    time.sleep(2) # let the camera focus
    
    photo_count = 0
    
    print(f"Taking photos for {subject}.")
    
    while True:
        #  opencv expects an array of image data (height x width x color_channels) 
        frame = picam.capture_array()
        # show the live feed of the camera 
        cv2.imshow("Capture", frame)
        # each 1 milisecond chekc waht key was pressed 
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(' '): 
            photo_count += 1
            timestamp = dt.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{subject}_{timestamp}.jpg"
            filepath = os.path.join(folder, filename)
            cv2.imwrite(filepath, frame) # write the array data to a JPG file 
            print(f"Photo {photo_count} saved to {filepath}")
            
        elif key == ord('q'):
            break
        
    cv2.destroyAllWindows()
    picam.stop()
    print(f"PHOTO CAPTURE COMPLETE. {photo_count} photos saved for {subject}")
          
          
        
        
if __name__ == "__main__":
    capture_photos(person_name)