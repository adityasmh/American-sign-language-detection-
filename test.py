import cv2
import mediapipe as mp
import pickle
import numpy as np
import pyttsx3
import time
import nltk
from nltk.corpus import brown
from nltk import bigrams, ConditionalFreqDist
from collections import deque

# Build Bigram model
words = brown.words()
word_pairs = list(bigrams([w.lower() for w in words]))
cfd = ConditionalFreqDist(word_pairs)

# Load trained SVM model
model_dict = pickle.load(open('model.p', 'rb'))
model = model_dict['model']

# Mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Text-to-speech
engine = pyttsx3.init()
engine.setProperty('rate', 100)

# Initialize
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Error: Could not open webcam.")
    exit()

finalized_text = ""
last_backspace_time = 0
last_clear_time = 0
hand_x_history = deque(maxlen=5)  # For swipe smoothing

with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        H, W, _ = frame.shape
        frame = cv2.flip(frame, 1)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        frame_output = frame.copy()

        data_aux = []
        x_, y_ = [], []
        gesture = None

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame_output,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(28, 255, 3), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)
                )
                wrist = hand_landmarks.landmark[0]
                curr_hand_x = wrist.x
                hand_x_history.append(curr_hand_x)

                # Smooth swipe detection
                if len(hand_x_history) == hand_x_history.maxlen:
                    dx = hand_x_history[-1] - hand_x_history[0]
                    if dx < -0.3 and (time.time() - last_backspace_time > 1.5):
                        finalized_text = finalized_text[:-1]
                        last_backspace_time = time.time()
                        print("⬅️  Swipe Left: Backspace")

                for lm in hand_landmarks.landmark:
                    data_aux.append(lm.x - wrist.x)
                    data_aux.append(lm.y - wrist.y)
                    data_aux.append(lm.z - wrist.z)
                    x_.append(lm.x)
                    y_.append(lm.y)

            if len(data_aux) == 63:
                prediction = model.predict([np.array(data_aux)])[0]
                gesture = prediction

                x1 = int(min(x_) * W) - 10
                y1 = int(min(y_) * H) - 10
                x2 = int(max(x_) * W) + 10
                y2 = int(max(y_) * H) + 10

                cv2.rectangle(frame_output, (x1, y1 - 10), (x2, y2), (255, 99, 173), 4)
                cv2.putText(frame_output, prediction, (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 2)

        # Display finalized text
        cv2.putText(frame_output, f"Text: {finalized_text}", (10, H - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Gesture actions
        current_time = time.time()
        if gesture == 'clear' and current_time - last_clear_time > 2:
            finalized_text = ""
            last_clear_time = current_time

        # Keyboard controls
        key = cv2.waitKey(10) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('f'):
            if gesture and gesture not in ['clear', 'backspace']:
                finalized_text += gesture
        elif key == ord('b'):
            finalized_text = finalized_text[:-1]
        elif key == ord(' '):
            finalized_text += ' '
        elif key == ord('t'):
            engine.say(finalized_text)
            engine.runAndWait()

        cv2.imshow('ASL Virtual Keyboard', frame_output)

cap.release()
cv2.destroyAllWindows()
