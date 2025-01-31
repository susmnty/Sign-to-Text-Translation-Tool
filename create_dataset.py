import os
import pickle
import cv2
import mediapipe as mp

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Directory where the data is stored
DATA_DIR = './data'

# Data and label storage
data = []
labels = []

# Process images in the directory
for dir_ in os.listdir(DATA_DIR):
    class_dir = os.path.join(DATA_DIR, dir_)

    if os.path.isdir(class_dir):  # Process only the directories for each class (A-Z)
        print(f'Processing images for class {dir_}')

        for img_path in os.listdir(class_dir):
            img = cv2.imread(os.path.join(class_dir, img_path))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Process the image to detect hand landmarks
            results = hands.process(img_rgb)

            data_aux = []
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    x_ = []
                    y_ = []

                    # Extract hand landmarks and normalize them
                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        x_.append(x)
                        y_.append(y)

                    # Normalize landmarks
                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        data_aux.append(x - min(x_))
                        data_aux.append(y - min(y_))

                # Append the extracted data and the corresponding label (class)
                data.append(data_aux)
                labels.append(dir_)

# Save the dataset as a pickle file
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print("Dataset created and saved as 'data.pickle'")
