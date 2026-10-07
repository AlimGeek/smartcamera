# Smart Hand Gesture & Face Tracking System 

A real-time computer vision project built with Python, OpenCV, and MediaPipe. This system tracks hand gestures to recognize specific commands and applies dynamic AR filters based on user interaction

## Features 
* **Independent Multi-Hand Tracking:** Tracks and analyzes the left and right hands simultaneously
* **Complex Gesture Recognition:** Accurately recognizes states like "Peace," "Middle Finger," "Thumbs Up/Down," and counts individual fingers, ignoring natural physiological variations
* **Gesture-Triggered AR Filters:** Automatically activates a face-tracking AR filter (dynamic GIF overlay) when showing two "Peace" signs
* **Built-in Screen Recording:** Press 'R' to toggle recording the output to an `.avi` file

## Tech Stack 
* Python 3.11
* OpenCV (`opencv-python`)
* MediaPipe (`mediapipe==0.10.21`)
* Pillow

## Installation & Setup 🚀
 Clone the repository:

   git clone [https://github.com/YOUR_GITHUB_USERNAME/YOUR_REPOSITORY_NAME.git](https://github.com/YOUR_GITHUB_USERNAME/YOUR_REPOSITORY_NAME.git)
   cd YOUR_REPOSITORY_NAME

 sync the environment using uv:
  uv sync

 Run the application:
  uv run forgit.py

 Controls 
  Show ✌️  with both hands to activate the AR filter.
  Press R to start/stop video recording.
  Press Q to quit.

 Author 
  Alim
  Undergraduate Student in IME at XJTU