import os
import cv2
import numpy as np

# Paths to the files
weights_path = "yolov3.weights"
cfg_path = "yolov3.cfg"
names_path = "coco.names"

# Verify that the files exist
if not os.path.exists(weights_path):
    raise FileNotFoundError("YOLOv3 weights file not found. Please download it and place it in the current directory.")

if not os.path.exists(cfg_path):
    raise FileNotFoundError("YOLOv3 configuration file not found. Please download it and place it in the current directory.")

if not os.path.exists(names_path):
    raise FileNotFoundError("COCO names file not found. Please download it and place it in the current directory.")

# Load YOLO
net = cv2.dnn.readNet(weights_path, cfg_path)
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load COCO names
with open(names_path, "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Load image
img = cv2.imread("newyork.jpg")
height, width, channels = img.shape

# Detecting objects
blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob)
outs = net.forward(output_layers)

# Showing information on the screen
class_ids = []
confidences = []
boxes = []

for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            # Object detected
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            # Rectangle coordinates
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

font = cv2.FONT_HERSHEY_PLAIN
colors = np.random.uniform(0, 255, size=(len(classes), 3))

for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        color = colors[class_ids[i]]
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, label, (x, y - 10), font, 1, color, 2)

# Save the output image
output_path = "output.jpg"
cv2.imwrite(output_path, img)
print(f"Output image saved to {output_path}")
