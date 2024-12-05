import os
import mediapipe as mp
import cv2
import pickle
import numpy as np

# Load model
model_dict = pickle.load(open('model.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize Mediapipe hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.9)

finalized_text = ""  # Variable to store the finalized text

with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8) as hands:
    while cap.isOpened():

        data_aux = []
        x_ = []
        y_ = []

        ret, frame = cap.read()
        H, W, _ = frame.shape

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb = cv2.flip(frame_rgb, 1)
        frame_rgb.flags.writeable = False
        results = hands.process(frame_rgb)
        frame_rgb.flags.writeable = True
        frame_rgb = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame_rgb,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(28, 255, 3), thickness=5, circle_radius=10),
                    mp_drawing.DrawingSpec(color=(236, 255, 3), thickness=5, circle_radius=10)
                )

            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    data_aux.append(hand_landmarks.landmark[i].x)
                    data_aux.append(hand_landmarks.landmark[i].y)

            prediction = model.predict([np.array(data_aux)[0:42]])[0]
            cv2.putText(frame_rgb, prediction, (50, 50), cv2.FONT_HERSHEY_DUPLEX, 2, (255, 0, 0), 2, cv2.LINE_AA)

        # Display the finalized text at the bottom-left corner
        cv2.putText(frame_rgb, f"Text: {finalized_text}", (10, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                    cv2.LINE_AA)

        cv2.imshow('frame', frame_rgb)

        key = cv2.waitKey(10) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('f'):  # Press 'f' to add the current prediction to the text
            finalized_text += prediction
        elif key == ord('b'):  # Press 'b' to remove the last character
            finalized_text = finalized_text[:-1]

cap.release()
cv2.destroyAllWindows()