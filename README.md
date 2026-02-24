🎯 Multimodal Maestro: Human Gaze & Gesture Based HCI System
  An AI-powered, hands-free Human-Computer Interaction system enabling cursor control through eye gaze tracking and hand gesture recognition — designed for both general users and individuals with mobility impairments.

📄 Published: IEEE COMPSIF 2025 | Springer ICAIH (CCIS Series)
🏫 Institution: PES Institute of Technology and Management, Shivamogga (VTU)
👥 Team: Ananya R, Dhanya M Bhat, Nireeksha B, Akanksha S

📌 Overview
Multimodal Maestro allows users to control their computer without a keyboard or mouse by using:
  👁️ Eye Gaze — move the cursor using eye movements; blink to click
  🖐️ Hand Gestures — control cursor and trigger actions using finger movements
  🌐 Web Interface — switch between both modes in real time via a Flask-powered web app
This system is especially meaningful for users with paralysis or limited mobility, providing an inclusive, accessible computing experience.

✨ Key Features	
 👁️ Gaze-based Cursor Control:	Real-time cursor movement via eye landmark tracking
 😉 Blink Detection:	Eye blinks trigger mouse clicks (EAR-based detection)
 🖐️ Gesture-based Control:	Finger gestures map to click, copy, paste, select-all
 🔄 Modality Switching: Switch between gaze and gesture via web UI
 📷 Real-time Processing:	Low-latency webcam-based input processing
 ♿ Accessibility Focus:	Designed for mobility-impaired users
 
🧠 Models & Algorithms
Gaze Recognition (Objective 1)
  VGG-19 + LSTM hybrid for blink classification (spatial + temporal)
  MediaPipe FaceMesh — 468 facial landmark detection
  EAR (Eye Aspect Ratio) — blink detection trigger
  CNN — eye state classification | Accuracy: 89.85%
  Image preprocessing: resize to 128×128, normalization, augmentation (flip, rotate, contrast)
Hand Gesture Recognition (Objective 2)
  CNN-LSTM hybrid — spatial + temporal gesture modeling
  MobileNetV2 — lightweight real-time gesture classification
  MediaPipe Hands — 21-landmark hand tracking
  Gesture model accuracy: 82.69%
  Adam optimizer + weighted cross-entropy loss
Web Application (Objective 3)
  Flask backend to start/stop controllers via HTTP POST
  HTML, CSS, JavaScript frontend with tab navigation
  Real-time feedback and error handling
  
🛠️ Tech Stack
Computer Vision   : OpenCV, MediaPipe (FaceMesh + Hands)
Deep Learning     : TensorFlow, Keras (VGG-19, LSTM, MobileNetV2, CNN)
Automation        : PyAutoGUI (cursor control, click simulation)
Web Backend       : Flask (Python)
Web Frontend      : HTML5, CSS3, JavaScript
Data Processing   : NumPy, Pillow
IDE               : VS Code, Google Colab

📁 Project Structure
multimodal-maestro-hci/
│
├── eye_tracker.py           # Gaze-based cursor control module
├── Finger_tracking.py       # Hand gesture cursor control module
├── train.ipynb              # Model training notebook (VGG-19 + LSTM, CNN-LSTM)
├── app/
│   ├── templates/           # HTML frontend
│   └── static/              # CSS, JS files
├── models/                  # Saved trained model files
├── requirements.txt
└── README.md

⚙️ Installation & Setup
1. Clone the Repository
git clone https://github.com/NireekshaB/multimodal-maestro-hci.git
cd multimodal-maestro-hci
2. Install Dependencies
pip install -r requirements.txt
3. Requirements
opencv-python
mediapipe
pyautogui
tensorflow
numpy
pillow
flask
4. Run Modules
Eye Tracker (Gaze Control):
python eye_tracker.py

Hand Gesture Control:
python Finger_tracking.py

Web Application:
python app.py
# Open http://localhost:5000 in your browser

🎮 How to Use
Gaze Mode
  Look at the screen — cursor follows your eye movement
  Blink to perform a mouse click
  Works with standard webcam (no special hardware needed)
  
Gesture Mode
Gesture	                    Action
Thumb + Index finger tap	  Left click
Thumb + Middle finger tap	  Select All (Ctrl+A)
Thumb + Ring finger tap	    Copy (Ctrl+C)
Thumb + Little finger tap	  Paste (Ctrl+V)
Index finger movement	      Cursor movement

📊 Results
Model	                          Accuracy
CNN (Eye Blink Detection)	      89.85%
LSTM	                          80.12%
MobileNetV2	                    84.40%
VGG-16	                        75.30%
VGG-19	                        75.16%
CNN-LSTM (Gesture Recognition)	82.69%

📄 Publication
Title: "Navigating Interfaces: Gaze and Gesture as Multimodal Inputs"
Conference: IEEE COMPSIF 2025 | Springer ICAIH (CCIS Series)
Published: April 28, 2025 on IEEE Xplore
Keywords: HCI, CNN, LSTM, Computer Vision, Accessibility, EAR, MediaPipe

👥 Team
Name	         USN
Ananya R	     4PM21AI004
Dhanya M Bhat	 4PM21AI012
Nireeksha B	   4PM21AI024
Akanksha S	   4PM21AI057
Guide: Mr. Sandeep Telkar R, Assistant Professor, Dept. of AI&ML, PESITM

📜 License
This project was developed as a final year B.E. project at PESITM, Shivamogga under VTU, Belagavi.

⭐ If you found this project interesting, please consider giving it a star!
