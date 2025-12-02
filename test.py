import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os
from glob import glob
from collections import defaultdict
import math
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns

# Load saved model and scaler (assuming retrained on new features)
clf = joblib.load('ballet_rf_model2.pkl')
scaler = joblib.load('ballet_rf_scaler2.pkl')

# Adjust paths for test data (assume similar structure to training, e.g., test/first/, test/second/ with JSONs)
test_jsons_root = r'./test_data_output'  # Change to your test JSONs folder
output_test_csv = 'test_ballet_features_rf2.csv'  # Optional: Save extracted test features

BODY_25_INDICES = {
    'LHip': 12, 'RHip': 9,
    'LKnee': 13, 'RKnee': 10,
    'LAnkle': 14, 'RAnkle': 11,
    'LBigToe': 19, 'LSmallToe': 20, 'LHeel': 21,
    'RBigToe': 22, 'RSmallToe': 23, 'RHeel': 24
}
FOOT_INDICES = [19,20,21,22,23,24]

def get_point(kps, idx, min_conf=0.3):
    x, y, c = kps[idx*3:idx*3+3]
    return (x, y, c) if c > min_conf else None

def compute_angle(p1, p2, p3):
    if None in (p1, p2, p3): return 0
    v1 = (p1[0]-p2[0], p1[1]-p2[1])
    v2 = (p3[0]-p2[0], p3[1]-p2[1])
    dot = v1[0]*v2[0] + v1[1]*v2[1]
    mag1, mag2 = math.sqrt(v1[0]**2 + v1[1]**2), math.sqrt(v2[0]**2 + v2[1]**2)
    return math.degrees(math.acos(dot / (mag1 * mag2 + 1e-6))) if mag1 > 0 and mag2 > 0 else 0

def extract_features_from_video(json_files):
    frame_features = []
    valid_frame_count = 0
    for json_file in sorted(json_files):
        with open(json_file, 'r') as f:
            data = json.load(f)
        if not data['people']: continue

        # Merge keypoints (highest conf)
        merged_kps = [0] * 75
        for person in data['people']:
            kps = person['pose_keypoints_2d']
            for i in range(75):
                if kps[i] > merged_kps[i]:
                    merged_kps[i] = kps[i]

        # Get points (lower conf for legs to 0.3)
        points = {name: get_point(merged_kps, idx, 0.3) for name, idx in BODY_25_INDICES.items()}

        # Filter: Require key legs (relaxed)
        if None in (points['LKnee'], points['LAnkle'], points['RKnee'], points['RAnkle']):
            continue

        # Foot points with lower conf (0.2)
        foot_points = [p for idx in FOOT_INDICES if (p := get_point(merged_kps, idx, 0.2))]

        if len(foot_points) < 4:
            continue

        # Compute features
        lankle, rankle = points['LAnkle'], points['RAnkle']
        lknee, rknee = points['LKnee'], points['RKnee']

        left_leg_len = math.dist(lknee[:2], lankle[:2])
        right_leg_len = math.dist(rknee[:2], rankle[:2])
        avg_leg_len = (left_leg_len + right_leg_len) / 2

        # Updated cross_factor: 1 if left over right (left front), -1 if right over left (right front), 0 otherwise
        if lankle[0] > rankle[0]:
            cross_factor = 1
        elif rankle[0] > lankle[0]:
            cross_factor = -1
        else:
            cross_factor = 0

        # Visibility counts for feet
        left_foot_vis = sum(1 for name in ['LBigToe', 'LSmallToe', 'LHeel'] if points[name] is not None)
        right_foot_vis = sum(1 for name in ['RBigToe', 'RSmallToe', 'RHeel'] if points[name] is not None)

        # Back heel to front toe distances (based on cross direction)
        if cross_factor == 1:  # Left front, right back
            back_heel_to_front_bigtoe = math.dist(points['RHeel'][:2], points['LBigToe'][:2]) / avg_leg_len if points['RHeel'] and points['LBigToe'] and avg_leg_len > 0 else 0
            back_heel_to_front_smalltoe = math.dist(points['RHeel'][:2], points['LSmallToe'][:2]) / avg_leg_len if points['RHeel'] and points['LSmallToe'] and avg_leg_len > 0 else 0
        elif cross_factor == -1:  # Right front, left back
            back_heel_to_front_bigtoe = math.dist(points['LHeel'][:2], points['RBigToe'][:2]) / avg_leg_len if points['LHeel'] and points['RBigToe'] and avg_leg_len > 0 else 0
            back_heel_to_front_smalltoe = math.dist(points['LHeel'][:2], points['RSmallToe'][:2]) / avg_leg_len if points['LHeel'] and points['RSmallToe'] and avg_leg_len > 0 else 0
        else:  # No clear cross, default to 0
            back_heel_to_front_bigtoe = 0
            back_heel_to_front_smalltoe = 0

        features = {
            'ankle_dist': math.dist(lankle[:2], rankle[:2]),
            'foot_spread': max(p[0] for p in foot_points) - min(p[0] for p in foot_points),
            'foot_y_std': np.std([p[1] for p in foot_points]) if foot_points else 0,
            'left_straightness': abs(lknee[0] - lankle[0]),
            'right_straightness': abs(rknee[0] - rankle[0]),
            'left_turnout_angle': compute_angle(points['LHip'], lankle, points['LBigToe']),
            'right_turnout_angle': compute_angle(points['RHip'], rankle, points['RBigToe']),
            'cross_factor': cross_factor,
            'left_leg_len': left_leg_len,
            'right_leg_len': right_leg_len,
            'heel_toe_overlap_left': math.dist(points['LHeel'][:2], points['RBigToe'][:2]) / avg_leg_len if points['LHeel'] and points['RBigToe'] and avg_leg_len > 0 else 0,
            'heel_toe_overlap_right': math.dist(points['RHeel'][:2], points['LSmallToe'][:2]) / avg_leg_len if points['RHeel'] and points['LSmallToe'] and avg_leg_len > 0 else 0,
            'ankle_x_dist': abs(lankle[0] - rankle[0]) / avg_leg_len if avg_leg_len > 0 else 0,
            'left_foot_vis': left_foot_vis,
            'right_foot_vis': right_foot_vis,
            'back_heel_to_front_bigtoe': back_heel_to_front_bigtoe,
            'back_heel_to_front_smalltoe': back_heel_to_front_smalltoe,
        }
        # Normalize by avg leg len
        if avg_leg_len > 0:
            features['ankle_dist'] /= avg_leg_len
            features['foot_spread'] /= avg_leg_len

        frame_features.append(features)
        valid_frame_count += 1

    if not frame_features:
        print(f"Skipped video {json_files[0]}: no valid frames")  # Debug skip
        return None
    print(f"Video {json_files[0]}: {valid_frame_count}/{len(json_files)} valid frames")  # Debug count
    # Average over frames
    avg_features = {k: np.mean([f[k] for f in frame_features]) for k in frame_features[0]}
    return avg_features

# Extract features from test data
data = []
position_labels = {'first': 0, 'second': 1, 'third': 2, 'fourth': 3, 'fifth': 4}  # For true labels if folders are named

for pos_name, true_label in position_labels.items():
    pos_dir = os.path.join(test_jsons_root, pos_name)
    if not os.path.exists(pos_dir): continue

    video_jsons = defaultdict(list)
    for json_file in glob(os.path.join(pos_dir, '*_keypoints.json')):
        base = os.path.basename(json_file).rsplit('_keypoints.json', 1)[0]
        parts = base.rsplit('_', 1)
        if len(parts) == 2:
            prefix = parts[0]
            video_jsons[prefix].append(json_file)

    for prefix, json_files in video_jsons.items():
        features = extract_features_from_video(json_files)
        if features:
            features['true_label'] = true_label  # For evaluation
            features['video_id'] = prefix
            data.append(features)

test_df = pd.DataFrame(data)
test_df.to_csv(output_test_csv, index=False)  # Optional save

# Prepare for prediction (drop non-features)
test_features = test_df.drop(['true_label', 'video_id'], axis=1)
true_labels = test_df['true_label']

# Scale and predict
test_features_scaled = scaler.transform(test_features)
preds = clf.predict(test_features_scaled)

# Evaluate
print(f'Accuracy on test data: {accuracy_score(true_labels, preds):.2f}')
print(classification_report(true_labels, preds))

# Output predictions per video
test_df['predicted_label'] = preds
print(test_df[['video_id', 'true_label', 'predicted_label']])

# Confusion matrix for better misclassification insight
cm = confusion_matrix(true_labels, preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=position_labels.keys(), yticklabels=position_labels.keys())
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Your scatterplot (updated to use new features if desired; here using ankle_x_dist for better relevance to changes)
test_df['error'] = test_df['true_label'] != test_df['predicted_label']  # Flag errors
sns.scatterplot(data=test_df, x='ankle_x_dist', y='foot_spread', hue='true_label', style='error')
plt.title('Feature Scatter: True Labels with Errors Marked (Using New ankle_x_dist)')
plt.show()  # This will pop up a plot window ``