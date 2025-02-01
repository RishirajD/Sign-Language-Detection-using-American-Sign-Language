import cv2
import numpy as np
import time
from tensorflow.keras.models import load_model
from datetime import datetime
import threading

# Load the pretrained model
MODEL_PATH = './models/asl_inception_v3_trainedpro.h5'  # Path to your model
model = load_model(MODEL_PATH)

# Define class labels (A-Z)
class_labels = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z'
]

# Function to preprocess the ROI for the model
def preprocess_frame(frame):
    frame_resized = cv2.resize(frame, (299, 299), interpolation=cv2.INTER_LINEAR)  # Optimized resize
    frame_array = np.array(frame_resized, dtype=np.float16)  
    frame_array = frame_array / 255.0  
    frame_array = np.expand_dims(frame_array, axis=0)  
    return frame_array

# Initialize webcam with higher resolution
cap = cv2.VideoCapture(0)  # Use 0 for default webcam

# Set the webcam resolution (if supported)
cap.set(3, 1280)  # Set width to 1280
cap.set(4, 720)   # Set height to 720

# Check if the webcam is opened
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Press 'q' to quit. Press 'Enter' to add the word to the sentence. Press 'Space' to add a space between words.")

# Variables for FPS calculation
prev_time = 0
fps = 0

# Gesture history and sentence builder
gesture_history = []
current_word = []  
sentence = ""  # To store the final sentence
last_prediction = None  
confidence_threshold = 0.75  # Minimum confidence for valid prediction
debounce_time = 1  # Minimum time (in seconds) between predictions
last_prediction_time = time.time()

# Capture frame in a separate thread for non-blocking operation
def capture_frame():
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        return None
    return cv2.flip(frame, 1)  # Flip horizontally for mirror effect

# Main loop to process frames
while True:
    # Capture a frame using a non-blocking function
    frame = capture_frame()
    if frame is None:
        continue

    # Get current frame dimensions
    screen_height, screen_width = frame.shape[:2]

    # Dynamically calculate ROI based on screen size (right side of the screen)
    roi_width = int(screen_width * 0.3)  # 30% of screen width
    roi_height = int(screen_height * 0.5)  # 50% of screen height
    roi_start_x = screen_width - roi_width - 20  # Padding from the right
    roi_start_y = 20  # Padding from the top
    roi_end_x = roi_start_x + roi_width
    roi_end_y = roi_start_y + roi_height

    # Add a semi-transparent light blue overlay for the ROI
    overlay = frame.copy()
    cv2.rectangle(overlay, (roi_start_x, roi_start_y), (roi_end_x, roi_end_y), (255, 200, 150), -1)  # Light blue rectangle
    alpha = 0.3  # Transparency factor
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    # Extract the ROI for gesture detection
    roi = frame[roi_start_y:roi_end_y, roi_start_x:roi_end_x]

    # Preprocess the ROI
    preprocessed_roi = preprocess_frame(roi)

    # Make prediction
    predictions = model.predict(preprocessed_roi)
    predicted_confidence = np.max(predictions)
    predicted_class = np.argmax(predictions)
    predicted_label = class_labels[predicted_class]

    # Apply confidence threshold and debounce mechanism
    if predicted_confidence >= confidence_threshold and time.time() - last_prediction_time > debounce_time:
        last_prediction_time = time.time()
        last_prediction = predicted_label

        # Add the prediction to the gesture history (keep last 5)
        if len(gesture_history) >= 5:
            gesture_history.pop(0)
        gesture_history.append(predicted_label)

        # Update the current word
        if len(current_word) == 0 or (current_word and current_word[-1] != predicted_label):
            current_word.append(predicted_label)

    # Detect key presses for word management
    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):  # Space key to add a space in the sentence
        if current_word:
            word = ''.join(current_word)
            sentence += word + " "
            current_word = []
    elif key == 13:  # Enter key to finalize the current word into the sentence
        if current_word:
            word = ''.join(current_word)
            sentence += word + " "
            current_word = []

    # Display current word and sentence at the bottom of the screen
    cv2.rectangle(frame, (20, screen_height - 140), (screen_width - 20, screen_height - 20), (0, 0, 0), -1)  # Black background
    cv2.putText(frame, f'Current Word: {"".join(current_word)}', (30, screen_height - 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, f'Sentence: {sentence.strip()}', (30, screen_height - 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Display the gesture history on the left side of the ROI
    history_start_x = 20  # Place history at the top-left
    history_start_y = 20
    for i, gesture in enumerate(gesture_history):
        cv2.putText(frame, f"{i + 1}. {gesture}", (history_start_x, history_start_y + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)  # Black text for gesture history

    # Calculate and display FPS at the bottom of the ROI
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if curr_time - prev_time > 0 else 0
    prev_time = curr_time
    cv2.putText(frame, f'FPS: {int(fps)}', (roi_start_x + 10, roi_end_y + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)  # Yellow FPS display

    # Display current datetime dynamically at the top-right corner of the screen
    datetime_text = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    cv2.putText(frame, datetime_text, (screen_width - 300, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Show the live video feed with predictions
    cv2.imshow("ASL Recognition", frame)

    # Break loop on 'q' key press
    if key == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
