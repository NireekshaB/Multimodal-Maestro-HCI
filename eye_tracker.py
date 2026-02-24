import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import os
import shutil

# Create directories to store eye images
dataset_directory = 'eye_img'

if not os.path.exists(dataset_directory):
    os.makedirs(dataset_directory)

# Initialize Mediapipe and OpenCV
cam = cv2.VideoCapture(0)  # Capture from the first video source
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
screen_width, screen_height = pyautogui.size()  # Get screen dimensions

# Parameters for smoothing cursor movement
prev_x, prev_y = screen_width / 2, screen_height / 2
smooth_factor = 0.1  # Decrease smooth factor for less rapid movement

# Blink detection thresholds
EYE_AR_THRESH = 0.004
EYE_AR_CONSEC_FRAMES = 3
blink_counter = 0
image_counter = 0
blink_image_counter = 0

# Helper function to calculate eye aspect ratio
def eye_aspect_ratio(eye_landmarks):
    if len(eye_landmarks) < 6:
        return 0  # Return 0 if there are not enough landmarks
    vertical_dist = eye_landmarks[1].y - eye_landmarks[5].y
    horizontal_dist = eye_landmarks[2].x - eye_landmarks[0].x
    return vertical_dist / horizontal_dist

# Helper function to apply smoothing
def smooth_cursor(prev_x, prev_y, target_x, target_y, smooth_factor):
    smooth_x = prev_x + (target_x - prev_x) * smooth_factor
    smooth_y = prev_y + (target_y - prev_y) * smooth_factor
    return smooth_x, smooth_y

# Helper function to draw landmarks on the frame
def draw_landmarks(frame, landmarks):
    for landmark in landmarks:
        x, y = int(landmark.x * frame_width), int(landmark.y * frame_height)
        cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)  # Draw landmarks as green circles

# Initialize lists to store real data
pupil_landmarks = []
eye_blinks = []

while True:  # Run forever
    _, frame = cam.read()  # Read the frame from the camera
    frame = cv2.flip(frame, 1)  # Flip frame horizontally
    frame_height, frame_width, _ = frame.shape  # Get frame dimensions

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert frame to RGB
    output = face_mesh.process(rgb_frame)  # Process the RGB frame

    landmark_points = output.multi_face_landmarks

    # Initialize eye_frame to a default value
    eye_frame = np.zeros_like(frame)

    if landmark_points:  # If face landmarks are detected
        landmarks = landmark_points[0].landmark  # Choose the first detected face

        # Draw landmarks on frame
        draw_landmarks(frame, landmarks)

        # Extract relevant eye landmarks (4 key points for each eye: inner, outer, top, bottom)
        left_eye_landmarks = [landmarks[33], landmarks[133], landmarks[159], landmarks[145]]  # Inner, outer, top, bottom of left eye
        right_eye_landmarks = [landmarks[362], landmarks[263], landmarks[386], landmarks[374]]  # Inner, outer, top, bottom of right eye

        # Calculate bounding box for both eyes using 4 key points
        def get_eye_bbox(eye_landmarks):
            x_coords = [landmark.x for landmark in eye_landmarks]
            y_coords = [landmark.y for landmark in eye_landmarks]
            x_min = int(min(x_coords) * frame_width)
            x_max = int(max(x_coords) * frame_width)
            y_min = int(min(y_coords) * frame_height)
            y_max = int(max(y_coords) * frame_height)
            return (x_min, y_min, x_max, y_max)

        left_eye_bbox = get_eye_bbox(left_eye_landmarks)
        right_eye_bbox = get_eye_bbox(right_eye_landmarks)

        # Combine bounding boxes to encompass both eyes
        min_x = min(left_eye_bbox[0], right_eye_bbox[0])
        min_y = min(left_eye_bbox[1], right_eye_bbox[1])
        max_x = max(left_eye_bbox[2], right_eye_bbox[2])
        max_y = max(left_eye_bbox[3], right_eye_bbox[3])

        # Ensure bounding box is within frame limits
        min_x = max(0, min_x - 20)
        min_y = max(0, min_y - 20)
        max_x = min(frame_width, max_x + 20)
        max_y = min(frame_height, max_y + 20)

        # Crop frame to the bounding box of the eyes
        eye_frame = frame[min_y:max_y, min_x:max_x]

        # Cursor tracking (on original frame, not cropped)
        for id, landmark in enumerate(left_eye_landmarks + right_eye_landmarks):
            x = int(landmark.x * frame_width)  # Get position of face in width
            y = int(landmark.y * frame_height)  # Get position of face in height

            if id == 1:  # Assuming this landmark is the center for cursor tracking
                screen_x = int(landmark.x * screen_width)  # Get position of cursor along width
                screen_y = int(landmark.y * screen_height)  # Get position of cursor along height

                # Smooth cursor movement
                smooth_x, smooth_y = smooth_cursor(prev_x, prev_y, screen_x, screen_y, smooth_factor)
                # Add a movement threshold to avoid rapid jumps
                if abs(smooth_x - prev_x) > 5 or abs(smooth_y - prev_y) > 5:
                    pyautogui.moveTo(smooth_x, smooth_y)
                    prev_x, prev_y = smooth_x, smooth_y

        # Detect blink (close eyes)
        left_eye_distance = eye_aspect_ratio(left_eye_landmarks)
        right_eye_distance = eye_aspect_ratio(right_eye_landmarks)
        eye_distance = (left_eye_distance + right_eye_distance) / 2
        if eye_distance < EYE_AR_THRESH:
            blink_counter += 1
            if blink_counter >= EYE_AR_CONSEC_FRAMES:
                pyautogui.click()  # Perform click
                blink_counter = 0  # Reset blink counter

                # Append real data
                left_pupil_coords = (int(landmarks[468].x * frame_width), int(landmarks[468].y * frame_height))
                right_pupil_coords = (int(landmarks[473].x * frame_width), int(landmarks[473].y * frame_height))
                pupil_landmarks.append((left_pupil_coords, right_pupil_coords))
                eye_blinks.append(1)
        else:
            blink_counter = 0  # Reset blink counter if eyes are not closed

        # Save general eye images
        image_counter += 1
        general_eye_image = cv2.resize(eye_frame, (100, 100))  # Resize for consistency
        image_filename = os.path.join(dataset_directory, f"eye_{image_counter}.png")
        cv2.imwrite(image_filename, general_eye_image)
        print(f"Saved general eye image: {image_filename}")

        eye_blinks.append(0)

    # Display the cropped frame (only eyes) and the original frame
    cv2.imshow('Eye Frame', eye_frame)  # Show only eyes
    cv2.imshow('Full Frame', frame)  # Show full frame
    if cv2.waitKey(1) & 0xFF == 27:  # Exit on 'ESC' key
        break

cam.release()
cv2.destroyAllWindows()

# Sorting function to arrange images in order
def sort_images_in_folder(folder):
    # List all files in the directory
    files = os.listdir(folder)
    # Filter out non-image files
    image_files = [f for f in files if f.endswith('.png')]
    # Sort files numerically based on the number in the filename
    image_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))

    # Move files to a new directory in sorted order
    sorted_folder = os.path.join(folder, 'sorted')
    if not os.path.exists(sorted_folder):
        os.makedirs(sorted_folder)

    for idx, file_name in enumerate(image_files):
        src_path = os.path.join(folder, file_name)
        dst_path = os.path.join(sorted_folder, f"{idx + 1}.png")  # Renaming to 1.png, 2.png, ...
        shutil.move(src_path, dst_path)
        print(f"Moved {file_name} to {dst_path}")

# Run the sorting function after image collection
sort_images_in_folder(dataset_directory)

# Print out final collected real data
print("Pupil Landmarks Collected:", pupil_landmarks)
print("Eye Blink Data Collected:", eye_blinks)
