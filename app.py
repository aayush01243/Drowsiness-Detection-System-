from flask import Flask, render_template, Response, request
import cv2
import dlib
from scipy.spatial import distance
import numpy as np
import pygame
import time

app = Flask(__name__)

pygame.mixer.init()
alert_sound = pygame.mixer.Sound('C:/Users/ayush/OneDrive/Desktop/Final Year Project/DDS/Danger Alarm Sound Effect.mp3')

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('C:/Users/ayush/OneDrive/Desktop/Final Year Project/DDS/shape_predictor_68_face_landmarks.dat')

EAR_THRESHOLD = 0.25
CONSECUTIVE_FRAMES = 20
frame_counter = 0
drowsy_start_time = None
alert_active = False

# Global variables for contrast and brightness
contrast = 1.5
brightness = 0

def eye_aspect_ratio(eye):
    eye = np.array([(p.x, p.y) for p in eye])
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def generate_frames():
    global frame_counter, drowsy_start_time, alert_active, contrast, brightness
    cap = cv2.VideoCapture(0)
    alarm_video = cv2.VideoCapture('C:/Users/ayush/OneDrive/Desktop/Final Year Project/DDS/static/css/video.mp4')

    while True:
        if alert_active:
            ret, frame = alarm_video.read()
            if not ret:
                alarm_video.set(cv2.CAP_PROP_POS_FRAMES, 0)
                alert_active = False
                continue
        else:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (640, 480))
            frame = cv2.convertScaleAbs(frame, alpha=contrast, beta=brightness)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)

            for face in faces:
                landmarks = predictor(gray, face)
                left_eye = [landmarks.part(i) for i in range(36, 42)]
                right_eye = [landmarks.part(i) for i in range(42, 48)]
                left_ear = eye_aspect_ratio(left_eye)
                right_ear = eye_aspect_ratio(right_eye)
                ear = (left_ear + right_ear) / 2.0

                if ear < EAR_THRESHOLD:
                    frame_counter += 1
                    if frame_counter >= CONSECUTIVE_FRAMES:
                        if drowsy_start_time is None:
                            drowsy_start_time = time.time()
                        elapsed_time = time.time() - drowsy_start_time

                        if elapsed_time >= 3:
                            message = "WARNING: DROWSINESS DETECTED!"
                            cv2.putText(frame, message, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                            if not pygame.mixer.get_busy():
                                alert_sound.play()
                            alert_active = True
                else:
                    frame_counter = 0
                    drowsy_start_time = None
                    alert_active = False
                    message = "You are safe!"
                    cv2.putText(frame, message, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()
    alarm_video.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Routes to adjust contrast and brightness
@app.route('/adjust_contrast', methods=['POST'])
def adjust_contrast():
    global contrast
    value = float(request.args.get('value', 0))
    contrast += value
    return ('', 204)  # Return a 204 No Content response

@app.route('/adjust_brightness', methods=['POST'])
def adjust_brightness():
    global brightness
    value = int(request.args.get('value', 0))
    brightness += value
    return ('', 204)

if __name__ == '__main__':
    app.run(debug=True)
