import numpy as np

cm = np.array([
    [139, 39, 17, 42, 68, 171, 15],
    [5, 31, 1, 6, 2, 10, 0],
    [68, 23, 64, 36, 72, 203, 62],
    [40, 17, 8, 560, 68, 167, 19],
    [21, 25, 4, 57, 290, 216, 13],
    [34, 21, 16, 45, 99, 371, 8],
    [26, 8, 18, 34, 43, 49, 238]
])

emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
mood_map = {
    'Happy': 'Happy', 'Surprise': 'Happy',
    'Angry': 'Energetic', 'Fear': 'Energetic',
    'Sad': 'Sad', 'Disgust': 'Sad',
    'Neutral': 'Neutral'
}

total = cm.sum()
direct_correct = np.trace(cm)

thayer_correct = 0
for true_idx, true_emo in enumerate(emotions):
    true_mood = mood_map[true_emo]
    for pred_idx, pred_emo in enumerate(emotions):
        if mood_map[pred_emo] == true_mood:
            thayer_correct += cm[true_idx, pred_idx]

print(f"Total test samples: {total}")
print(f"Direct Accuracy: {direct_correct/total:.1%}")
print(f"Thayer Accuracy: {thayer_correct/total:.1%}")