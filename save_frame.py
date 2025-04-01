import numpy as np
import cv2
import os
from datetime import datetime

imagedir = "calib_images"

# Start camera with autofocus off to prevent focal length
cap = cv2.VideoCapture(0)
success = cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
#print(success)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    cv2.imshow('frame', frame)
    
    if cv2.waitKey(1) == ord('s'):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"frame_{timestamp}.jpg"
        filepath = os.path.join(imagedir, filename)
        cv2.imwrite(filepath, frame)
    
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
