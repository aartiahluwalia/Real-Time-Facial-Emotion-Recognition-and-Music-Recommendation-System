import base64
import numpy as np
import cv2
import keras
import pandas as pd
from flask import Flask, render_template, request, jsonify, session
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import pickle

app = Flask(__name__)
app.secret_key = "final_project_secure_key"


face_model = keras.models.load_model('finetuned_model.keras')

music_db = pd.read_csv('labeled_music_data.csv')

with open('music_mood_modell.pkl', 'rb') as f:
    music_model = pickle.load(f)

FEATURES = ['energy', 'valence', 'tempo', 'danceability', 'acousticness', 'instrumentalness']
music_db['mood'] = music_model.predict(music_db[FEATURES])

face_net = cv2.dnn.readNetFromCaffe(
    "deploy.prototxt",
    "res10_300x300_ssd_iter_140000.caffemodel"
)

EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral','Sad', 'Surprise']

MOOD_MAP = {
    'Happy': 'Happy', 'Surprise': 'Happy',
    'Angry': 'Energetic', 'Fear': 'Energetic',
    'Sad': 'Sad', 'Disgust': 'Sad',
    'Neutral': 'Neutral'
}

def detect_face(frame):
    h, w = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(
        frame, 1.0, (300, 300),
        (104.0, 177.0, 123.0)
    )

    face_net.setInput(blob)
    detections = face_net.forward()

    if detections.shape[2] == 0:
        return None

    i = np.argmax(detections[0, 0, :, 2])
    confidence = detections[0, 0, i, 2]

    if confidence < 0.5:
        return None

    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
    x1, y1, x2, y2 = box.astype(int)

    return x1, y1, x2 - x1, y2 - y1


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_mood', methods=['POST'])
def predict_mood():
    try:
        data = request.json['image']
        encoded_data = data.split(',')[1]
        nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        face_box = detect_face(frame)
        if face_box is None:
            return jsonify({'success': False, 'error': 'No face detected. Try better lighting or adjust position.'})

        x, y, w, h = face_box
        face_img = frame[y:y+h, x:x+w]

        face_img = cv2.resize(face_img, (96, 96))
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        img_array = np.expand_dims(face_img, axis=0)
        img_preprocessed = preprocess_input(img_array.astype('float32'))

        preds = face_model.predict(img_preprocessed)
        detected_emotion = EMOTIONS[np.argmax(preds)]
        target_mood = MOOD_MAP.get(detected_emotion, 'Neutral')

        session['detected_mood'] = target_mood

        return jsonify({
            'success': True,
            'mood': target_mood,
            'emotion': detected_emotion
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/playlist')
def playlist():
    user_mood = session.get('detected_mood', 'Neutral')
    filtered_songs = music_db[music_db['mood'] == user_mood]

    sample_size = min(len(filtered_songs), 10)
    suggestions = filtered_songs.sample(n=sample_size)
    songs_to_render = suggestions[['track_name', 'track_artist']].to_dict('records')

    return render_template('playlist.html', mood=user_mood, songs=songs_to_render)


if __name__ == '__main__':
    app.run(debug=True)

