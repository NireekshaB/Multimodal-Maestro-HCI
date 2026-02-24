import cv2
import mediapipe as mp
import pyautogui
import os
import math

# Create directory to store finger images
finger_dataset_directory = 'finger_datasets'

if not os.path.exists(finger_dataset_directory):
    os.makedirs(finger_dataset_directory)

# Initialize Mediapipe Hands and OpenCV
cam = cv2.VideoCapture(0)
hands = mp.solutions.hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
screen_width, screen_height = pyautogui.size()  # Get screen dimensions

# Parameters for smoothing cursor movement
prev_x, prev_y = screen_width / 2, screen_height / 2
smooth_factor = 0.3  # Increase smoothing factor for more stable movement

# Finger dataset image counter
finger_image_counter = 0

# Helper function to apply smoothing
def smooth_cursor(prev_x, prev_y, target_x, target_y, smooth_factor):
    smooth_x = prev_x + (target_x - prev_x) * smooth_factor
    smooth_y = prev_y + (target_y - prev_y) * smooth_factor
    return smooth_x, smooth_y

# Helper function to calculate distance between two points
def calculate_distance(point1, point2):
    return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

# Click threshold for thumb and finger taps
CLICK_THRESHOLD = 0.04  # Adjust for responsive click detection

# Action mapping
def perform_action(finger_combination, hand_frame):
    global finger_image_counter
    if finger_combination == 'middle':
        pyautogui.hotkey('ctrl', 'a')  # Select all
        print("Select All")
    elif finger_combination == 'ring':
        pyautogui.hotkey('ctrl', 'c')  # Copy
        print("Copy")
    elif finger_combination == 'little':
        pyautogui.hotkey('ctrl', 'v')  # Paste
        print("Paste")
    elif finger_combination == 'click':
        pyautogui.click()  # Perform a mouse click
        print("Mouse Click")

    # Save the image after the action is performed
    finger_image_counter += 1
    image_filename = os.path.join(finger_dataset_directory, f"finger_action_{finger_image_counter}.png")
    cv2.imwrite(image_filename, hand_frame)
    print(f"Saved action image: {image_filename}")

while True:
    _, frame = cam.read()
    frame = cv2.flip(frame, 1)
    frame_height, frame_width, _ = frame.shape
    
    # Convert frame to RGB for Mediapipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = hands.process(rgb_frame)  # Process the frame for hand landmarks

    hand_landmarks = output.multi_hand_landmarks

    if hand_landmarks:
        for handLms in hand_landmarks:
            # Get bounding box coordinates for the hand
            hand_x_min = min(lm.x for lm in handLms.landmark)
            hand_x_max = max(lm.x for lm in handLms.landmark)
            hand_y_min = min(lm.y for lm in handLms.landmark)
            hand_y_max = max(lm.y for lm in handLms.landmark)

            # Convert to pixel coordinates
            hand_x_min_pixel = int(hand_x_min * frame_width)
            hand_x_max_pixel = int(hand_x_max * frame_width)
            hand_y_min_pixel = int(hand_y_min * frame_height)
            hand_y_max_pixel = int(hand_y_max * frame_height)

            # Ensure the cropped region is valid before proceeding
            if hand_x_min_pixel >= 0 and hand_y_min_pixel >= 0 and hand_x_max_pixel <= frame_width and hand_y_max_pixel <= frame_height:
                hand_frame = frame[hand_y_min_pixel:hand_y_max_pixel, hand_x_min_pixel:hand_x_max_pixel]
                
                # Only save the hand frame if it's not empty
                if hand_frame.size > 0:
                    # Track the index finger tip (landmark 8), thumb tip (landmark 4), and other fingers
                    index_finger_tip = handLms.landmark[8]
                    thumb_tip = handLms.landmark[4]
                    middle_finger_tip = handLms.landmark[12]
                    ring_finger_tip = handLms.landmark[16]
                    little_finger_tip = handLms.landmark[20]

                    # Get screen coordinates for the cursor
                    screen_x = int(index_finger_tip.x * screen_width)
                    screen_y = int(index_finger_tip.y * screen_height)

                    # Smooth cursor movement
                    smooth_x, smooth_y = smooth_cursor(prev_x, prev_y, screen_x, screen_y, smooth_factor)
                    pyautogui.moveTo(smooth_x, smooth_y)
                    prev_x, prev_y = smooth_x, smooth_y

                    # Check for thumb + index finger tap to perform click
                    action_performed = False  # Track if an action was performed
                    if calculate_distance(thumb_tip, index_finger_tip) < CLICK_THRESHOLD:
                        perform_action('click', hand_frame)  # Trigger mouse click
                        action_performed = True
                    elif calculate_distance(thumb_tip, middle_finger_tip) < CLICK_THRESHOLD:
                        perform_action('middle', hand_frame)  # Select All
                        action_performed = True
                    elif calculate_distance(thumb_tip, ring_finger_tip) < CLICK_THRESHOLD:
                        perform_action('ring', hand_frame)  # Copy
                        action_performed = True
                    elif calculate_distance(thumb_tip, little_finger_tip) < CLICK_THRESHOLD:
                        perform_action('little', hand_frame)  # Paste
                        action_performed = True

                    # Draw hand landmarks on the frame
                    for id, lm in enumerate(handLms.landmark):
                        cx, cy = int(lm.x * frame_width), int(lm.y * frame_height)
                        cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)

    # Display the frame with zoomed-in hand
    cv2.imshow('Hand Tracking', frame)
    if cv2.waitKey(1) & 0xFF == 27:  # Exit on 'ESC' key
        break

cam.release()
cv2.destroyAllWindows()
