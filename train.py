import os
import mediapipe as mp
import cv2
import matplotlib.pyplot as plt
import pickle
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Setup mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.9)

# Load data
data_dir = ".venv/asl_alphabet_train"
data = []
labels = []

for label in sorted(os.listdir(data_dir)):
    if label == '.DS_Store':
        continue
    for img_file in os.listdir(os.path.join(data_dir, label)):
        data_aux = []
        img = cv2.imread(os.path.join(data_dir, label, img_file))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                wrist = hand_landmarks.landmark[0]
                for lm in hand_landmarks.landmark:
                    data_aux.append(lm.x - wrist.x)
                    data_aux.append(lm.y - wrist.y)
                    data_aux.append(lm.z - wrist.z)
            data.append(data_aux)
            labels.append(label)

# Save data
with open('.venv/data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

# Visualize one sample per label
for label in sorted(os.listdir(data_dir)):
    if label == '.DS_Store':
        continue
    img_file = os.listdir(os.path.join(data_dir, label))[0]
    img = cv2.imread(os.path.join(data_dir, label, img_file))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                img_rgb,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
    plt.figure()
    plt.title(label)
    plt.imshow(img_rgb)

plt.show()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(np.array(data), labels, test_size=0.15, random_state=22, shuffle=True)

# Train stronger model (SVM)
model = SVC(kernel='rbf', C=10, gamma='scale', probability=True)
model.fit(X_train, y_train)

# Evaluate
pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred))

# Save model
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)

# Load model
with open('model.p', 'rb') as f:
    model_dict = pickle.load(f)
model = model_dict['model']

with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)


# Real-time prediction
cap = cv2.VideoCapture(0)

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
                wrist = hand_landmarks.landmark[0]
                for lm in hand_landmarks.landmark:
                    data_aux.append(lm.x - wrist.x)
                    data_aux.append(lm.y - wrist.y)
                    data_aux.append(lm.z - wrist.z)
                    x_.append(lm.x)
                    y_.append(lm.y)

            if len(data_aux) == 63:  # Ensure only one hand detected
                x1 = int(min(x_) * W) - 10
                y1 = int(min(y_) * H) - 10
                x2 = int(max(x_) * W) + 10
                y2 = int(max(y_) * H) + 10

                prediction = model.predict([np.array(data_aux)])[0]

                cv2.rectangle(frame_rgb, (x1, y1 - 10), (x2, y2), (255, 99, 173), 6)
                cv2.putText(frame_rgb, prediction, (x1, y1), cv2.FONT_HERSHEY_DUPLEX, 5, (255, 0, 0), 5, cv2.LINE_AA)

        cv2.imshow('ASL Real-Time Detection', frame_rgb)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
