# Sign-to-Text Translation Tool

## Objective

The **Sign-to-Text Translation Tool** bridges the communication gap by translating hand sign language into readable text. It assists mute individuals by converting hand gestures into alphabets (A-Z) with high accuracy.

---

## Project Overview

This project is divided into four main components for modularity and clarity:

1. **Data Preparation (`data.pickle`)**  
   Handles structured storage of dataset information.
2. **Image Collection (`images_collection.py`)**  
   Facilitates systematic data collection for training models.
3. **Training Classifier (`train_classifier.py`)**  
   Processes data, trains a RandomForest model, and evaluates its performance.
4. **Inference Classifier (`inference_classifier.py`)**  
   Performs real-time hand gesture recognition.

---

## Components

### 1. Data Preparation
- **File Structure:**  
  A `data.pickle` file stores a dictionary with:  
  - **`data`:** The primary dataset for training.  
  - **`labels`:** Classification labels.  
- **Purpose:**  
  Acts as a compact representation of the dataset and its attributes for seamless processing.

---

### 2. Image Collection
- **Purpose:**  
  Captures image datasets for recognizing alphabets (A-Z).  
- **Key Features:**  
  - Creates a `data` folder with subfolders for each alphabet (A-Z).  
  - Captures 100 images per alphabet using a webcam (configurable).  
  - Prompts user interaction to start/stop data collection.  
- **Dependencies:**  
  Requires `OpenCV` and `os` libraries.

---

### 3. Training Classifier
- **Workflow:**  
  1. Preprocesses data by extracting and normalizing hand landmarks.  
  2. Trains a **RandomForestClassifier** using the processed data.  
  3. Saves the trained model as `model.pickle`.  
- **Key Requirements:**  
  - Python libraries: OpenCV, Mediapipe, Scikit-learn, Pickle.  
  - Clear, labeled gesture images.  
  - A functional webcam for testing.  
- **Steps to Run:**  
  1. Place the dataset in the `./data` folder.  
  2. Execute the script and select:  
     - Option 1: Preprocess Data  
     - Option 2: Train Classifier  
     - Option 3: Real-Time Detection  
  3. Press `Esc` to exit detection.

---

### 4. Inference Classifier
- **Purpose:**  
  Performs real-time hand gesture recognition using the trained model.  
- **Key Features:**  
  - **Model Loading:** Loads `model.pickle` for predictions.  
  - **Hand Tracking:** Uses Mediapipe to detect 21 hand-knuckle landmarks.  
  - **Prediction:** Classifies gestures into alphabets (A-Z).  
  - **Visualization:** Displays results with bounding boxes and predictions.  
- **Dependencies:**  
  - Libraries: OpenCV, Mediapipe, Numpy, Pickle.  
  - Hardware: Webcam for real-time input.  
- **Usage:**  
  Run the script and interact using the "q" key to quit.

---

## Challenges Faced

- Managing and integrating multiple components of the project.  
- Ensuring precise hand landmark detection and consistent datasets.  
- Overcoming inaccuracies to achieve a final accuracy of 99%.

---

## Future Enhancements

- Expand support to recognize complete words and phrases.  
- Improve processing speed and accuracy.  
- Extend functionality to integrate with robotic systems for automated communication.

---

## Hand-Landmark Example

### Sign Language Alphabet Representation
![Hand Landmark](https://img.freepik.com/premium-vector/hand-gesture-language-alphabet_23-2147881973.jpg)

### Mediapipe Hand Landmarks
![Mediapipe Hand Landmarks](https://mediapipe.dev/images/mobile/hand_landmarks.png)

---

## Additional Learning

- **OpenCV:** A library for computer vision and image processing.  
  [Learn more](https://www.youtube.com/watch?v=7irSQuL24qY)  
- **Mediapipe:** Framework for building ML-based pipelines for hand tracking.  
  [Learn more](https://www.youtube.com/watch?v=VDCdWwldlx4)  
- **Hand Landmark Model:** Detects 21 key points for precise gesture recognition.  
  [More info](https://google.github.io/mediapipe/solutions/hands.html)

---

## References

- [Face Detection, Face Mesh, OpenPose, Holistic, Hand Detection Using Mediapipe](https://www.youtube.com/watch?v=VDCdWwldlx4)  
- [Introduction to OpenCV](https://www.youtube.com/watch?v=7irSQuL24qY)

---

## Folder Structure

```plaintext
Sign-to-Text/
│
├── data/                      # Dataset for training
├── models/                    # Saved trained models (e.g., model.pickle)
├── images_collection.py       # Script for collecting image datasets
├── train_classifier.py        # Script for training the model
├── inference_classifier.py    # Real-time recognition script
├── data.pickle                # Preprocessed dataset
└── README.md                  # Project documentation


