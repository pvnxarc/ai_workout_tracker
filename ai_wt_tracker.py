#opencv(open source computer vision) used for computer vision and image processing tasks. Used in survelliance, self driving cars, medical imagining, robotics.
#Mediapipe is a framework developed by google. Provides solutions for detecting and tracking objects in images or video streams, like hands, face, bodies and more.
import cv2 as cv       #work with camera and create a window
import mediapipe as mp #helps us detect body positions
import numpy as np     #for mathematical calculations
from enum import Enum 

# Update ExerciseType enum
class ExerciseType(Enum):
    CURL = 1
    NECK_ROTATION = 2
    LATERAL_RAISES = 3
    TRICEPS_KICKBACK = 4
    SQUATS = 5

class ExerciseTracker:
    def __init__(self):
        # MediaPipe setup
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        
        # Exercise tracking variables
        self.counter = 0
        self.stage = None
        self.current_exercise = ExerciseType.CURL

    def calculate_angle(self, a, b, c):
        """Calculate angle between three points"""
        a = np.array(a)  #first (Shoulder point)
        b = np.array(b)  #mid (Elbow point)
        c = np.array(c)  #End (Wrist)
        
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        
        if angle > 180.0: #calculations for joints
             360 - angle
        return angle if angle <= 180 else 360 - angle

    def track_curl(self, landmarks):
        """Track dumbbell curl exercise"""
        shoulder = [
            landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
            landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
        ]
        elbow = [
            landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
            landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].y
        ]
        wrist = [
            landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].x,
            landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].y
        ]
        
        # Calculate angle
        angle = self.calculate_angle(shoulder, elbow, wrist)
        
        # Curl detection logic
        if angle > 160:
            self.stage = "down"
        if angle < 30 and self.stage == "down":
            self.stage = "up"
            self.counter += 1
        
        return angle

    def track_neck_rotation(self, landmarks):
        """Track neck rotation exercise"""
        nose = [
            landmarks[self.mp_pose.PoseLandmark.NOSE.value].x,
            landmarks[self.mp_pose.PoseLandmark.NOSE.value].y
        ]
        left_shoulder = [
            landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
            landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
        ]
        right_shoulder = [
            landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
            landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y
        ]
        
        # Calculate neck rotation angle
        neck_angle = self.calculate_angle(left_shoulder, nose, right_shoulder)
        
        # Neck rotation detection logic
        if neck_angle > 30 and neck_angle < 80:
            if self.stage != "rotated":
                self.stage = "rotated"
        elif neck_angle > 80 and self.stage == "rotated":
            self.counter += 1
            self.stage = "center"
        
        return neck_angle
        
    def track_lateral_raise(self, landmarks):
        """Track lateral raise exercise"""
        shoulder = [
            landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
            landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
        ]
        elbow = [
            landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
            landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].y
        ]
        wrist = [
            landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].x,
            landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].y
        ]
        
        # Calculate angle
        lateral_raise_angle = self.calculate_angle(shoulder, elbow, wrist)
        
        # Lateral raise detection logic
        if lateral_raise_angle > 160:
            self.stage = "lowered"
        
        if lateral_raise_angle < 30 and self.stage == "lowered":
            self.stage = "raised"
            self.counter += 1
        
        return lateral_raise_angle

    def track_triceps_kickback(self, landmarks):
        """Track triceps kickback exercise"""
        shoulder = [
            landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
            landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
        ]
        elbow = [
            landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
            landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].y
        ]
        wrist = [
            landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].x,
            landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].y
        ]
        
        # Calculate angle
        kickback_angle = self.calculate_angle(shoulder, elbow, wrist)
        
        # Triceps kickback detection logic
        if kickback_angle > 160:
        # Starting position - arm close to body
            self.stage = "initial"
    
        if kickback_angle < 90 and self.stage == "initial":
        # Extended position - arm pushed back
            self.stage = "extended"
            self.counter += 1
    
    def track_squats(self, landmarks):
        """Track squat exercise"""
        hip = [
        landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].x,
        landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y
        ]
        knee = [
        landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].x,
        landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].y
        ]
        ankle = [
        landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
        landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].y
        ]
    
        # Calculate squat angle
        squat_angle = self.calculate_angle(hip, knee, ankle)
    
        # Squat detection logic
        if squat_angle > 160:
            # Standing position
            self.stage = "standing"
    
        if squat_angle < 90 and self.stage == "standing":
        # Lowest point of squat
            self.stage = "squatted"
            self.counter += 1
    
        return squat_angle

def main():
    # Open video capture
    v_capture = cv.VideoCapture(0)
    
    # Create exercise tracker
    tracker = ExerciseTracker()
    
    # Print exercise switching instructions
    print("\n=== Exercise Tracker ===")
    print("Press keys to switch exercises:")
    print("1: Curl")
    print("2: Neck Rotation")
    print("3: Lateral Raises")
    print("4: Triceps Kickbacks")
    print("5: Squats")
    print("Q: Quit")
    
    with tracker.mp_pose.Pose(
        min_detection_confidence=0.5, 
        min_tracking_confidence=0.5
    ) as pose:
        while v_capture.isOpened():
            # Read frame
            ret, frame = v_capture.read()
            
            if not ret:
                break
            
            # Convert frame to RGB
            image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            image.flags.writeable = False
            
            # Process pose
            result = pose.process(image)
            
            # Convert back to BGR
            image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
            image.flags.writeable = True
            
            # Draw and process landmarks if detected
            if result.pose_landmarks:
                # Draw pose connections
                tracker.mp_drawing.draw_landmarks(
                    image, 
                    result.pose_landmarks, 
                    tracker.mp_pose.POSE_CONNECTIONS,
                    tracker.mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                    tracker.mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                )
                
                try:
                    landmarks = result.pose_landmarks.landmark
                    
                    # Track specific exercise
                    if tracker.current_exercise == ExerciseType.CURL:
                        angle = tracker.track_curl(landmarks)
                    elif tracker.current_exercise == ExerciseType.NECK_ROTATION:
                        angle = tracker.track_neck_rotation(landmarks)
                    elif tracker.current_exercise == ExerciseType.LATERAL_RAISES:
                        angle = tracker.track_lateral_raise(landmarks)
                    elif tracker.current_exercise == ExerciseType.TRICEPS_KICKBACK:
                        angle = tracker.track_triceps_kickback(landmarks)
                    elif tracker.current_exercise == ExerciseType.SQUATS:
                        angle = tracker.track_squats(landmarks)
                    
                    # Visualize angle
                    elbow = tracker.mp_pose.PoseLandmark.LEFT_ELBOW
                    cv.putText(
                        image, 
                        str(round(angle, 2)), 
                        tuple(np.multiply(
                            [landmarks[elbow.value].x, landmarks[elbow.value].y], 
                            [640, 480]
                        ).astype(int)),
                        cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv.LINE_AA
                    )
                except Exception as e:
                    print(f"Error tracking exercise: {e}")
            
            # Setup status box
            cv.rectangle(image, (0,0), (300,80), (245,117,16), -1)
            
            # Display exercise type
            cv.putText(image, f'Exercise: {tracker.current_exercise.name}', 
                       (10,30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            
            # Display rep count
            cv.putText(image, 'REPS', (15,12), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv.LINE_AA)
            cv.putText(image, str(tracker.counter), (10,60), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv.LINE_AA)
            
            # Display stage
            cv.putText(image, 'STAGE', (65,12), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv.LINE_AA)
            cv.putText(image, str(tracker.stage), (60,60), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv.LINE_AA)
            
            # Show frame
            cv.imshow('Exercise Tracker', image)
            
            # Handle key events
            key = cv.waitKey(10) & 0xFF
            
            if key == ord('1'):
                tracker.current_exercise = ExerciseType.CURL
                tracker.counter = 0
                tracker.stage = None
            elif key == ord('2'):
                tracker.current_exercise = ExerciseType.NECK_ROTATION
                tracker.counter = 0
                tracker.stage = None
            elif key == ord('3'):
                tracker.current_exercise = ExerciseType.LATERAL_RAISES
                tracker.counter = 0
                tracker.stage = None
            elif key == ord('4'):
                tracker.current_exercise = ExerciseType.TRICEPS_KICKBACK
                tracker.counter = 0
                tracker.stage = None
            elif key == ord('5'):
                tracker.current_exercise = ExerciseType.SQUATS
                tracker.counter = 0
                tracker.stage = None
            elif key == ord('q'):
                break
    
    # Cleanup
    v_capture.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()