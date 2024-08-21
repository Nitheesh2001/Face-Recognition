import streamlit as st
import cv2
import numpy as np
import face_recognition
import os
from PIL import Image

# Load known faces
def load_known_faces(directory):
    known_face_encodings = []
    known_face_names = []
    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image = face_recognition.load_image_file(f"{directory}/{filename}")
            face_encoding = face_recognition.face_encodings(image)[0]
            known_face_encodings.append(face_encoding)
            known_face_names.append(os.path.splitext(filename)[0])
    return known_face_encodings, known_face_names

# Face recognition function
def recognize_faces(frame, known_face_encodings, known_face_names):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        face_names.append(name)

    for (top, right, bottom, left), name in zip(face_locations, face_names):
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

    return frame

def main():
    st.title("Face Recognition using Streamlit")
    st.write("This app detects and recognizes faces in real-time using your webcam.")

    known_face_encodings, known_face_names = load_known_faces("known_faces")

    run = st.checkbox("Run Face Recognition")
    FRAME_WINDOW = st.image([])

    camera = cv2.VideoCapture(0)

    while run:
        ret, frame = camera.read()
        if not ret:
            st.error("Failed to capture video")
            break

        frame = recognize_faces(frame, known_face_encodings, known_face_names)
        FRAME_WINDOW.image(frame, channels="BGR")

    camera.release()

if __name__ == "__main__":
    main()
