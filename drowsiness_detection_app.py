import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist
import streamlit as st
from threading import Thread
from playsound import playsound
import time

# Initialize dlib's face detector and the facial landmark predictor
@st.cache_resource
def load_models():
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    return detector, predictor

detector, predictor = load_models()

# Function to calculate EAR (Eye Aspect Ratio)
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# Function to calculate MAR (Mouth Aspect Ratio)
def mouth_aspect_ratio(mouth):
    A = dist.euclidean(mouth[2], mouth[10])  # Vertical distance
    B = dist.euclidean(mouth[4], mouth[8])   # Horizontal distance
    return A / B

# Function to play alarm sound
def play_alarm():
    playsound("alarm.mp3")

# Streamlit App UI
st.title("Drowsiness Detection Application")
st.sidebar.header("Configuration")
EAR_THRESHOLD = st.sidebar.slider("EAR Threshold", 0.2, 0.4, 0.25)
MAR_THRESHOLD = st.sidebar.slider("MAR Threshold", 1.0, 2.0, 1.7)
TIME_THRESHOLD = st.sidebar.slider("Time Threshold (seconds)", 1, 5, 2)

st.sidebar.text("Press 'q' in the video frame to quit.")

# Video processing function
def process_video():
    # Initialize variables
    COUNTER = 0
    MAR_COUNTER = 0
    ALARM_ON = False
    current_alert = ""
    alarm_trigger_time = 0
    ear_values = []
    mar_values = []

    # Start video capture
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("Error: Could not access the camera.")
        return

    # Create Streamlit placeholder for video feed
    video_placeholder = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to grab frame. Exiting...")
            break

        frame = cv2.resize(frame, (800, 600))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        for face in faces:
            shape = predictor(gray, face)
            shape = np.array([[p.x, p.y] for p in shape.parts()])

            left_eye = shape[42:48]
            right_eye = shape[36:42]
            mouth = shape[48:68]

            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            ear = (left_ear + right_ear) / 2.0

            ear_values.append(ear)
            if len(ear_values) > 5:
                ear_values.pop(0)
            smoothed_ear = np.mean(ear_values)

            mar = mouth_aspect_ratio(mouth)
            mar_values.append(mar)
            if len(mar_values) > 5:
                mar_values.pop(0)
            smoothed_mar = np.mean(mar_values)

            if smoothed_ear < EAR_THRESHOLD:
                if COUNTER == 0:
                    alarm_trigger_time = time.time()
                COUNTER += 1
                if time.time() - alarm_trigger_time >= TIME_THRESHOLD and not ALARM_ON:
                    ALARM_ON = True
                    current_alert = "Drowsiness Detected!"
                    Thread(target=play_alarm).start()

            else:
                COUNTER = 0
                if ALARM_ON:
                    ALARM_ON = False
                    current_alert = ""

            if smoothed_mar > MAR_THRESHOLD:
                MAR_COUNTER += 1
                if MAR_COUNTER == 1:
                    alarm_trigger_time = time.time()
                if time.time() - alarm_trigger_time >= TIME_THRESHOLD and not ALARM_ON:
                    ALARM_ON = True
                    current_alert = "Yawning Detected!"
                    Thread(target=play_alarm).start()

            else:
                MAR_COUNTER = 0
                if ALARM_ON and current_alert == "Yawning Detected!":
                    ALARM_ON = False
                    current_alert = ""

            # Draw landmarks
            cv2.drawContours(frame, [left_eye], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [right_eye], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [mouth], -1, (0, 0, 255), 1)

            if current_alert:
                cv2.putText(frame, current_alert, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Display EAR and MAR values
        cv2.putText(frame, f"EAR: {smoothed_ear:.2f}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"MAR: {smoothed_mar:.2f}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Show frame in Streamlit
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(frame, channels="RGB")

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

# Streamlit button to start the app
if st.button("Start Drowsiness Detection"):
    process_video()
