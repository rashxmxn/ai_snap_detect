from ultralytics import YOLO
import cv2

model = YOLO("/home/beybars/Desktop/beybars/projects/side_hustle/EKTU/yolo9e.pt")  # provide path to the trained model 
results = model("/home/beybars/Desktop/beybars/projects/side_hustle/EKTU/test.jpg")  # Run inference
result = results[0]
# Get the original image
img = result.orig_img

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

# Plot the boxes
for box in result.boxes:
    # Get box coordinates
    x1, y1, x2, y2 = box.xyxy[0].detach().cpu().numpy()
    
    # Get class name and confidence
    class_id = int(box.cls[0].item())
    conf = box.conf[0].item()
    class_name = result.names[class_id]
    
    # Get color for this class
    color = COLORS.get(class_name, (0, 255, 0))  # Default to green if class not found
    
    # Convert coordinates to integers
    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
    
    # Draw rectangle with class-specific color
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    
    # Add label with same color
    label = f'{class_name} {conf:.2f}'
    cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Display the image
cv2.imshow('Detection Result', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Optionally save the image
cv2.imwrite('result.jpg', img)