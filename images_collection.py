import os
import cv2

# Directory where the data will be saved
DATA_DIR = './data'
os.makedirs(DATA_DIR, exist_ok=True)

# Number of classes (A and B)
number_of_classes = 2  # 'A' and 'B'
dataset_size = 100

# Open the webcam
cap = cv2.VideoCapture(0)

# Loop over the alphabets (A and B)
for j in range(number_of_classes):
    class_name = chr(65 + j)  # Convert 0 -> 'A', 1 -> 'B'
    class_dir = os.path.join(DATA_DIR, class_name)
    os.makedirs(class_dir, exist_ok=True)

    print(f'Collecting data for class {class_name}')

    # Prompt user before starting collection
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from webcam")
            break
        cv2.putText(frame, f'Press "Q" to start collecting {class_name}!', (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Capture dataset_size images
    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from webcam")
            break
        cv2.imshow('frame', frame)
        cv2.waitKey(1)
        image_path = os.path.join(class_dir, f'{counter}.jpg')
        cv2.imwrite(image_path, frame)
        counter += 1
        print(f"Saved {image_path}")

cap.release()
cv2.destroyAllWindows()
