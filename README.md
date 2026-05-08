# Emotion-Driven Music Recommendation System

> A real-time web application that detects facial emotions through a webcam and recommends music matching the user's mood, built using deep learning and audio feature classification.

![Python](https://img.shields.io/badge/python-3.10-blue.svg)
![TensorFlow](https://img.shields.io/badge/tensorflow-2.15-orange.svg)
![Flask](https://img.shields.io/badge/flask-2.3-green.svg)
![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)
![Status](https://img.shields.io/badge/status-MSc%20final%20project-success.svg)

---

## Overview

This project bridges affective computing and music information retrieval by combining facial expression recognition with audio feature-based music classification. A user's webcam frame is processed through a multi-stage machine learning pipeline that detects faces, classifies emotion, and recommends a playlist whose mood matches the detected emotional state.

**Key technical components:**
- MobileNetV2-based facial emotion classifier (7 classes) trained on FER-2013
- OpenCV DNN SSD face detector with ResNet-10 backbone
- Random Forest music mood classifier on Spotify audio features
- Thayer's arousal-valence model for mapping emotions to music moods
- Flask web application with browser-based webcam capture

This was developed as a final MSc Computer Science project at the University of Hertfordshire (2026).

---

## How it works

Webcam frame → Face detection → Emotion classification → Mood mapping → Playlist
(browser)      (DNN SSD)        (MobileNetV2)        (Thayer)      (Random Forest)

1. The browser captures a webcam frame and sends it as a base64 JPEG to the Flask backend
2. OpenCV DNN SSD detects and crops the face region
3. The cropped face is preprocessed (96×96 RGB, MobileNetV2 normalisation) and passed to the emotion classifier
4. The predicted emotion is mapped to one of four Thayer mood quadrants (Happy, Energetic, Sad, Neutral)
5. A Random Forest classifier filters the song catalogue to that mood; ten songs are sampled and presented with YouTube search links

End-to-end inference takes approximately 500 ms on standard hardware.

---

## Project structure

emotion-music-recommendation/

├── app.py                              # Flask backend

├── facial_emotion_detection_f.ipynb  # FER model training notebook

├── preprocess_music_data.ipynb     # Music classifier training notebook

├── finetuned_model.keras           # Trained MobileNetV2 model (~26 MB)

├── music_mood_model.pkl            # Trained Random Forest classifier

├── labeled_music_data.csv          # Spotify song catalogue (without mood column)

├── deploy.prototxt                 # SSD face detector architecture

├── res10_300x300_ssd_iter_140000.caffemodel  # SSD face detector weights

├── templates/

│   ├── index.html                  # Capture page

│   └── playlist.html               # Recommendation page

├── requirements.txt

└── README.md


---

## Installation

### Prerequisites

- Python 3.10 or later
- pip (Python package manager)
- A working webcam
- A modern browser (Chrome, Firefox, or Edge)

### Step 1: Clone the repository

```bash
git clone https://github.com/yourusername/emotion-music-recommendation.git
cd emotion-music-recommendation
```

### Step 2: Create a virtual environment (recommended)

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Verify model files

Make sure these files are in the project root:
- `finetuned_model.keras`
- `music_mood_model.pkl`
- `labeled_music_data.csv`
- `deploy.prototxt`
- `res10_300x300_ssd_iter_140000.caffemodel`

If any are missing, regenerate them by running the notebooks (see "Retraining" below).

### Step 5: Run the application

```bash
python app.py
```

Open your browser at `http://127.0.0.1:5000` and grant webcam permission when prompted.

---

## Usage

1. Open the application in your browser
2. Click **"Allow"** when prompted for webcam access
3. Click **"Analyze My Face"** to capture a frame
4. Wait for the system to detect your emotion
5. Click **"Go to Playlist"** to see ten mood-matched song recommendations
6. Click any **"Play"** button to open a YouTube search for that song

If face detection fails, ensure good lighting and that your face is roughly centred and visible in the webcam preview.

---

## Retraining the models

### FER model
The training notebook downloads FER-2013 from KaggleHub and trains MobileNetV2 in two phases. **Note: full training takes 5-6 hours on CPU.**

```bash
jupyter notebook facial_emotion_detection_f.ipynb
```

### Music classifier
The music notebook combines two Spotify CSV datasets, applies the Thayer labelling rule, and trains a Random Forest. **Training takes about 2 minutes.**

```bash
jupyter notebook preprocess_music_data.ipynb
```

---

## Technical details

### FER model (MobileNetV2)
- **Architecture:** MobileNetV2 backbone + custom head (Dense 256 → BN → Dropout 0.4 → Dense 128 → BN → Dropout 0.3 → Dense 7 softmax)
- **Training:** Two-phase transfer learning — Phase 1 (30 epochs, frozen backbone, LR 1e-3), Phase 2 (20 epochs, last 30 layers unfrozen, LR 1e-5)
- **Input:** 96×96 RGB, preprocessed with MobileNetV2's `preprocess_input` (range [-1, 1])
- **Test accuracy:** 47.2% on 3,589 unseen FER-2013 images
- **Class weights:** Computed using sklearn's `balanced` strategy to address class imbalance (Disgust 9.41, Happy 0.57)

### Music classifier (Random Forest)
- **Dataset:** 4,830 songs combining high and low popularity Spotify tracks
- **Features:** energy, valence, tempo, danceability, acousticness, instrumentalness
- **Labelling:** Rule-based using Thayer's arousal-valence quadrants (threshold 0.5)
- **Trees:** 100 estimators, stratified 80/20 train-test split

### Face detection (OpenCV DNN SSD)
- **Architecture:** ResNet-10 SSD pre-trained on faces (`res10_300x300_ssd_iter_140000.caffemodel`)
- **Input:** 300×300 blob with mean subtraction (104, 177, 123)
- **Confidence threshold:** 0.5

---

## Hypotheses tested

**H1: Model achieves ≥60% accuracy on FER-2013 test set**
Result: **Not supported** — achieved 47.2%. Gap explained by CPU training, 96×96 input resolution, and FER-2013's known label noise (~30%).

**H2: Thayer mood mapping outperforms direct one-to-one emotion-to-playlist mapping**
Result: **Supported** — Thayer mood-level accuracy of 51.9% vs direct mapping accuracy of 47.2%, a 4.7 percentage point improvement.

---

## Limitations and future work

**Known limitations:**
- Strong over-prediction bias toward Sad class (33% of all predictions)
- 96×96 input resolution below MobileNetV2's intended 224×224
- CPU-only training restricted hyperparameter exploration
- No demographic bias evaluation across race, gender, or age

**Future improvements:**
- Focal loss to address class imbalance and Sad over-prediction
- AffectNet dataset (440K images) for cleaner labels
- GPU training, attention mechanisms, ensemble methods
- TensorFlow.js for client-side inference
- Spotify API integration for direct playback
- Mood regulation mode (sad emotion → uplifting music) with ethics-approved user study

---

## Ethical considerations

This system processes facial data, which UK GDPR classifies as biometric special-category data. The application:
- Processes frames in memory only — no images are stored or transmitted to external servers
- Activates the webcam only on explicit user click
- Does not perform identification or any persistent biometric tracking
- Does not log emotion data linked to identifiable users

For production deployment, formal privacy impact assessment, demographic bias evaluation, and ethics-approved user studies would be required.

---

## Author

**Aarti Ahluwalia**
MSc Computer Science | University of Hertfordshire | 2026
Supervisor: Dr. Chidinma Chiejina

---

## Acknowledgements

- FER-2013 dataset by Goodfellow et al. (2013)
- MobileNetV2 by Sandler et al. (2018) — Google
- Spotify Web API for audio feature data
- OpenCV DNN face detection model — Caffe community
- University of Hertfordshire MSc Computer Science programme

---

## License

This project is released under the MIT License. See `LICENSE` for details.
