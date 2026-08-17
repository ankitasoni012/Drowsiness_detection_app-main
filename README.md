Driver Drowsiness Detection System

The Driver Drowsiness Detection System is a real-time computer vision application designed to detect signs of driver fatigue, including drowsiness and yawning, and provide an alert to improve road safety.

Features

* **Drowsiness Detection** – Detects signs of fatigue using eye movement analysis.
* **Yawning Detection** – Identifies yawning using mouth movement analysis.
* **Facial Landmark Detection** – Uses Dlib's 68-point facial landmark model for facial feature detection.
* **Eye Aspect Ratio (EAR)** – Calculates eye aspect ratio to identify prolonged eye closure.
* **Mouth Aspect Ratio (MAR)** – Calculates mouth aspect ratio to detect yawning.
* **Audio Alert** – Triggers an alarm when signs of drowsiness are detected.
* **Streamlit Interface** – Provides a simple and user-friendly interface for running the application.

## Tech Stack

* **Python**
* **OpenCV**
* **Dlib**
* **Streamlit**
* **NumPy**

## How It Works

1. The application captures the driver's video input.
2. OpenCV detects the face from the video frames.
3. Dlib's 68-point facial landmark model identifies key facial landmarks.
4. The **Eye Aspect Ratio (EAR)** is calculated to monitor eye closure.
5. The **Mouth Aspect Ratio (MAR)** is calculated to identify yawning.
6. If the detected values remain beyond the defined thresholds, the system identifies signs of drowsiness.
7. An audio alert is triggered to warn the driver.

## Project Files

* `drowsiness_detection_app.py` – Main application code.
* `drowsiness_app.ipynb` – Jupyter Notebook containing the development and analysis work.
* `alarm.mp3` – Audio alert used by the application.

## Objective

The objective of this project is to develop a real-time driver monitoring system that can detect early signs of fatigue and provide timely alerts, helping reduce the risk of accidents caused by drowsy driving.

## Future Enhancements

* Improve detection accuracy under different lighting conditions.
* Add real-time driver monitoring and event logging.
* Improve alert mechanisms.
* Explore additional facial and behavioral indicators of fatigue.

## Author
**Ankita Soni**

