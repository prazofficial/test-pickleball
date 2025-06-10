import cv2
import mediapipe as mp

KITCHEN_LINE_Y = 300  # tune this for your camera angle

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, model_complexity=1)

def detect_foot_fault(frame):
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if not results.pose_landmarks:
        return "NO_PERSON"

    foot_fault = False
    for landmark_id in [mp_pose.PoseLandmark.LEFT_FOOT_INDEX, mp_pose.PoseLandmark.RIGHT_FOOT_INDEX]:
        landmark = results.pose_landmarks.landmark[landmark_id]
        y_px = int(landmark.y * frame.shape[0])
        if y_px >= KITCHEN_LINE_Y:
            foot_fault = True

    return "FOOT_FAULT" if foot_fault else "CLEAN"
