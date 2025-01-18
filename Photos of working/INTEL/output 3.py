import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Initialize YOLOv8 model
model = YOLO('yolov8s.pt')

# Path to the input video file
video_path = r'C:\Users\syed\OneDrive\Desktop\4\sample video.mp4'

# Open the video capture
cap = cv2.VideoCapture(video_path)

# Define codec and create VideoWriter for YOLO detection video
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output_detection.avi', fourcc, cap.get(cv2.CAP_PROP_FPS),
                      (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

# Variables for FPS calculation
fps_values = [10000]

# TTC calculation variables
prev_boxes = []
prev_frame_time = None
ttc_alert_lower = 0.5
ttc_alert_upper = 0.7

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Start timing for FPS calculation
    start_time = time.time()

    # YOLO detection
    results = model(frame)[0]
    current_boxes = []

    for result in results.boxes:
        box = result.xyxy[0]
        cls = int(result.cls[0])
        conf = result.conf[0]
        label = model.names[cls]

        # Filter only cars and high-confidence detections
        if label == 'car' and conf > 0.7:
            x1, y1, x2, y2 = map(int, box)
            current_boxes.append((x1, y1, x2, y2))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green bounding box
            cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    # Calculate TTC
    if prev_boxes and prev_frame_time:
        current_frame_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0  # Current time in seconds
        time_diff = current_frame_time - prev_frame_time

        for i, (x1, y1, x2, y2) in enumerate(current_boxes):
            if i < len(prev_boxes):
                prev_x1, prev_y1, prev_x2, prev_y2 = prev_boxes[i]

                # Calculate distances
                distance_current = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                distance_prev = np.sqrt((prev_x2 - prev_x1) ** 2 + (prev_y2 - prev_y1) ** 2)

                # Calculate relative speed (pixels per second)
                speed = (distance_prev - distance_current) / time_diff if time_diff > 0 else 0

                # Estimate TTC
                ttc = distance_current / speed if speed != 0 else float('inf')

                # Display TTC on the frame
                cv2.putText(frame, f'TTC: {ttc:.2f}s', (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

                # TTC Alert
                if ttc_alert_lower < ttc < ttc_alert_upper:
                    cv2.putText(frame, 'TTC ALERT!', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Update previous boxes and time for the next frame
    prev_boxes = current_boxes
    prev_frame_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

    # Calculate FPS and store for visualization
    fps_value = 1 / (time.time() - start_time)
    fps_values.append(fps_value)

    # Display FPS on the frame
    cv2.putText(frame, f'FPS: {fps_value:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # Write frame to output video
    out.write(frame)

    # Display the frame
    cv2.imshow('YOLO Detection with TTC', frame)

    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and writer
cap.release()
out.release()
cv2.destroyAllWindows()

# Plot performance metrics (FPS over time)
plt.figure(figsize=(10, 6))
plt.plot(fps_values, label='FPS', color='blue')
plt.title('FPS Performance Over Time')
plt.xlabel('Frame Index')
plt.ylabel('FPS')
plt.legend()
plt.grid()
plt.savefig('fps_performance.png')
plt.show()

print("YOLO Detection video saved as 'output_detection.avi'.")
print("FPS performance plot saved as 'fps_performance.png'.")
