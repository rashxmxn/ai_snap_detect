from ultralytics import YOLO
import cv2
import time

# Define colors for each class (in BGR format)
COLORS = {
    'person': (0, 0, 255),       # Red
    'ear': (0, 255, 0),         # Green
    'ear-mufs': (255, 0, 0),    # Blue
    'face': (255, 255, 0),      # Cyan
    'face-guard': (255, 0, 255), # Magenta
    'face-mask': (0, 255, 255),  # Yellow
    'foot': (128, 0, 0),        # Dark Blue
    'tool': (0, 128, 0),        # Dark Green
    'glasses': (0, 0, 128),     # Dark Red
    'gloves': (128, 128, 0),    # Dark Cyan
    'helmet': (128, 0, 128),    # Dark Magenta
    'hands': (0, 128, 128),     # Dark Yellow
    'head': (128, 128, 128),    # Gray
    'medical-suit': (64, 0, 0),  # Light Blue
    'shoes': (0, 64, 0),        # Light Green
    'safety-suit': (0, 0, 64),  # Light Red
    'safety-vest': (255, 128, 0) # Orange
}

def main():
    # Load YOLO model
    model = YOLO("/home/beybars/Desktop/beybars/projects/side_hustle/EKTU/yolo9c.pt")

    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    # Set camera resolution (optional)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Initialize FPS counter variables
    fps = 0
    frame_count = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Perform detection
        results = model(frame)
        result = results[0]

        # Calculate FPS
        frame_count += 1
        if frame_count >= 30:  # Update FPS every 30 frames
            end_time = time.time()
            fps = frame_count / (end_time - start_time)
            frame_count = 0
            start_time = time.time()

        # Draw FPS counter
        cv2.putText(frame, f'FPS: {fps:.1f}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Process and draw detections
        for box in result.boxes:
            # Get box coordinates
            x1, y1, x2, y2 = box.xyxy[0].detach().cpu().numpy()
            
            # Get class name and confidence
            class_id = int(box.cls[0].item())
            conf = box.conf[0].item()
            class_name = result.names[class_id]
            
            # Get color for this class
            color = COLORS.get(class_name, (0, 255, 0))
            
            # Convert coordinates to integers
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            
            # Draw rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Create background for text
            label = f'{class_name} {conf:.2f}'
            (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - 30), (x1 + label_width, y1), color, -1)
            
            # Add label
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Add info text
        cv2.putText(frame, 'Press "q" to quit', (20, frame.shape[0] - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Display the frame
        cv2.imshow('Real-time Object Detection', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 