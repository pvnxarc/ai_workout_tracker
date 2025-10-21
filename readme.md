# AI Workout Tracker
AI Workout Tracker is a computer vision-based fitness assistant that uses **OpenCV** and **MediaPipe** to track and analyze your exercises in real-time. It helps you maintain proper form, count repetitions, and stay motivated during your workouts.

## Features
1. Real-time exercise tracking using your webcam
2. Automatic rep counting for each exercise
3. Pose detection and guidance for better form
4. Lightweight and easy to use

## Supported Exercises
The tracker currently supports the following exercises:
1. **Curls** – Monitor arm curls for biceps growth.  
2. **Neck Rotation** – Track neck rotation movements for flexibility and mobility.  
3. **Lateral Raises** – Correctly perform lateral raises to strengthen shoulders.  
4. **Tricep Kickbacks** – Ensure proper form for tricep workouts.  
5. **Squats** – Monitor depth and form to protect your knees and maximize results.  

Installation
1. Clone the repository:
```bash
git clone https://github.com/yourusername/ai-workout-tracker.git
cd ai-workout-tracker
```
2. Install dependencies:
```bash
pip install opencv-python mediapipe
```
3. Run the tracker:
```bash
python ai_wt_tracker.py
```
Make sure web cam is connected and accessible.

## How It Works
OpenCV: Captures real-time video from your webcam and displays exercise feedback.
MediaPipe: Detects body landmarks to analyze movement and count repetitions.

