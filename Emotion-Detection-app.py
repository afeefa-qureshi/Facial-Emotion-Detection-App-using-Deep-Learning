import streamlit as st
import cv2
from keras.models import model_from_json
import numpy as np

# Load the pre-trained model architecture from JSON file
json_file = open("emotiondetector.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)

# Load the pre-trained model weights
model.load_weights("emotiondetector.h5")

# Load the Haar cascade classifier for face detection
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

# Define a function to extract features from an image
def extract_features(image):
    # Convert the color image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = np.array(gray_image)
    feature = feature.reshape(1, 48, 48, 1)  # Use 1 channel for grayscale image
    return feature / 255.0

# Define labels for emotion classes
labels = {0: 'angry', 1: 'fear', 2: 'happy', 3: 'neutral', 4: 'sad', 5: 'surprise'}

# Streamlit app
st.title("Facial Emotion Detection App using Deep Learning")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)

    # Detect faces in the color frame
    faces = face_cascade.detectMultiScale(image, 1.3, 5)

    try:
        # For each detected face, perform facial emotion recognition
        for (p, q, r, s) in faces:
            # Extract the region of interest (ROI) which contains the face
            face_image = image[q:q + s, p:p + r]

            # Resize the face image to the required input size (48x48)
            face_image = cv2.resize(face_image, (48, 48))

            # Extract features from the resized face image
            img = extract_features(face_image)

            # Make a prediction using the trained model
            pred = model.predict(img)

            # Get the predicted label for emotion
            prediction_label = labels[pred.argmax()]

            # Draw a rectangle around the detected face
            cv2.rectangle(image, (p, q), (p + r, q + s), (0, 0, 255), 8)

            # Display the predicted emotion label near the detected face with increased font size and red color
            cv2.putText(image, f'Emotion: {prediction_label}', (p, q + s + 30),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 10, (0, 0, 255), 10)

        # Display the frame with annotations in the Streamlit app
        st.image(image, channels="BGR", caption="Emotion Detection Result", use_column_width=True)

    except cv2.error:
        pass
