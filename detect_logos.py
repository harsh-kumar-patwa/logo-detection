from re import M
import json
import cv2
from ultralytics import YOLO

# Load the trained model
model_path = input("Enter the path to the trained model (e.g., 'path/to/your/model.pt'): ")
model = YOLO(model_path)

# Initialize video capture
video_path = input("Enter the path to the video file (e.g., 'path/to/your/video.mp4'): ")
video_capture = cv2.VideoCapture(video_path)

# Get frame rate and frame size of the video
frame_rate = video_capture.get(cv2.CAP_PROP_FPS)
frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
center_x = frame_width//2
center_y = frame_height//2

# Initialize video writer to save the output video
output_dir = input("Enter the directory to save the annotated video (e.g., 'path/to/your/directory'): ")
output_video_path = f"{output_dir}/output_video.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (frame_width, frame_height))

# Initialize lists to hold timestamps
pepsi_timestamps = []
cocacola_timestamps = []

current_frame_number = 0

while video_capture.isOpened():
    ret, frame = video_capture.read()
    if not ret:
        break

    # Perform detection
    detection_results = model(frame)

    # Process detections
    for result in detection_results:
        if hasattr(result, 'boxes') and result.boxes is not None:
            for box in result.boxes:
                class_label = result.names[int(box.cls)]
                confidence_score = box.conf
                bounding_box = box.xyxy[0].cpu().numpy()  # Get bounding box coordinates

                # Calculate size and distance from the center of the frame
                box_width = bounding_box[2] - bounding_box[0]
                box_height = bounding_box[3] - bounding_box[1]
                box_size = int(box_width * box_height)
                box_center_x = (bounding_box[0] + bounding_box[2]) // 2
                box_center_y = (bounding_box[1] + bounding_box[3]) // 2
                distance_from_center = int(((box_center_x - center_x) ** 2 + (box_center_y - center_y) ** 2) ** 0.5)

                # Calculate timestamp
                timestamp = current_frame_number / frame_rate

                if class_label == "Pepsi":  
                    pepsi_timestamps.append(timestamp)

                    # Draw bounding box and annotations on the frame
                    cv2.rectangle(frame, (int(bounding_box[0]), int(bounding_box[1])), (int(bounding_box[2]), int(bounding_box[3])), (255, 0, 0), 2)
                    cv2.putText(frame, f"{class_label}", (int(bounding_box[0]), int(bounding_box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                    cv2.putText(frame, f"Size: {box_size}", (int(bounding_box[0]), int(bounding_box[3]) + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                    cv2.putText(frame, f"Dist: {distance_from_center:.2f}", (int(bounding_box[0]), int(bounding_box[3]) + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

                elif class_label == "Cocacola":  
                    cocacola_timestamps.append(timestamp)

                    # Draw bounding box and annotations on the frame
                    cv2.rectangle(frame, (int(bounding_box[0]), int(bounding_box[1])), (int(bounding_box[2]), int(bounding_box[3])), (0, 0, 255), 2)
                    cv2.putText(frame, f"{class_label}", (int(bounding_box[0]), int(bounding_box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                    cv2.putText(frame, f"Size: {box_size}", (int(bounding_box[0]), int(bounding_box[3]) + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    cv2.putText(frame, f"Dist: {distance_from_center:.2f}", (int(bounding_box[0]), int(bounding_box[3]) + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Write the frame with bounding boxes to the output video
    video_writer.write(frame)
    current_frame_number += 1

# Release the video capture and writer objects
video_capture.release()
video_writer.release()

# Create the JSON structure
output_data = {
    "Pepsi": pepsi_timestamps,
    "Cocacola": cocacola_timestamps
}

# Write the JSON file
output_json_path = "/content/output_timestamps.json"
with open(output_json_path, "w") as json_file:
    json.dump(output_data, json_file, indent=4)

print(f"Timestamps have been saved to {output_json_path}")
print(f"Annotated video has been saved to {output_video_path}")
