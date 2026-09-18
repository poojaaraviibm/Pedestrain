import cv2
import numpy as np
import os
import winsound

# --- CONFIGURATION ---
CENTER_BOX_SIZE = (500, 800)  # Width, Height of the center zone
COLOR_SAFE = (255, 0, 0)      # Blue (BGR)
COLOR_ALERT = (0, 0, 255)     # Red (BGR)
CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4

def check_collision(box1, box2):
    """
    Check if two rectangles overlap (Axis-Aligned Bounding Box collision).
    box format: (x, y, w, h)
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    # Calculate top-left and bottom-right coordinates
    b1_x1, b1_y1 = x1, y1
    b1_x2, b1_y2 = x1 + w1, y1 + h1
    
    b2_x1, b2_y1 = x2, y2
    b2_x2, b2_y2 = x2 + w2, y2 + h2

    # Check for non-overlap conditions
    if (b1_x2 < b2_x1) or (b2_x2 < b1_x1) or (b1_y2 < b2_y1) or (b2_y2 < b1_y1):
        return False
    else:
        return True

# 1. Load YOLOv4
net = cv2.dnn.readNet("yolov4-tiny.weights", "yolov4-tiny.cfg")

with open("classes.txt", "r") as f:
    classes = [line.strip() for line in f.readlines()]
    
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# 2. Open Video Stream
cap = cv2.VideoCapture(0) # 0 for Webcam, or put 'video.mp4'

while True:
    ret, frame = cap.read()
    if not ret: break
    
    height, width, _ = frame.shape

    # --- DEFINE CENTER BOX COORDINATES ---
    cb_w, cb_h = CENTER_BOX_SIZE
    cb_x = int((width - cb_w) / 2)
    cb_y = int((height - cb_h) / 2)
    center_box_coords = (cb_x, cb_y, cb_w, cb_h)

    # 3. Detect Objects
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    # 4. Process YOLO Outputs
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # Only detect Class ID 0 (Person)
            if confidence > CONFIDENCE_THRESHOLD and classes[class_id] == "person":
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # 5. Non-Max Suppression
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    
    # 6. Check Collisions and Draw
    is_touching = False
    
    # Iterate through detected people to check collision FIRST
    if len(indexes) > 0:
        for i in indexes.flatten():
            person_box = boxes[i] # (x, y, w, h)
            if check_collision(center_box_coords, person_box):
                is_touching = True
                break # If one person touches, the box triggers

    # Determine Box Color
    current_color = COLOR_ALERT if is_touching else COLOR_SAFE
    thickness = 3 if is_touching else 2

    # Draw Center Box
    cv2.rectangle(frame, (cb_x, cb_y), (cb_x + cb_w, cb_y + cb_h), current_color, thickness)
    
    if is_touching:
        # os.system("say 'Pedestrian detected'")
        winsound.Beep(1000, 500)
        cv2.putText(frame, "WARNING: RESTRICTED AREA", (cb_x - 20, cb_y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, current_color, 2)

    # Draw People
    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            # Draw person bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "Person", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.imshow("Zone Detection", frame)
    
    key = cv2.waitKey(1)
    if key == 27: # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()