import json
import math
import numpy as np
import os
import pandas as pd
from glob import glob
from collections import defaultdict

# Adjust paths
data_root = r'./'  # Folders like first/, second/ (if used; optional)
jsons_root = r'./training_data_json2'  # Folders like first/, second/
output_csv = 'ballet_features2_2.csv'

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
        # Normalize
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

# Process all
data = []
position_labels = {'first': 0, 'second': 1, 'third': 2, 'fourth': 3, 'fifth': 4}

for pos_name, label in position_labels.items():
    pos_dir = os.path.join(jsons_root, pos_name)
    print(pos_dir)

    if not os.path.exists(pos_dir): continue

    # Group JSONs by prefix (video ID)
    video_jsons = defaultdict(list)
    for json_file in glob(os.path.join(pos_dir, '*_keypoints.json')):
        base = os.path.basename(json_file).rsplit('_keypoints.json', 1)[0]
        parts = base.rsplit('_', 1)  # Split off the frame number
        if len(parts) == 2:
            prefix = parts[0]
            video_jsons[prefix].append(json_file)

    # Process each video group
    for prefix, json_files in video_jsons.items():
        features = extract_features_from_video(json_files)
        if features:
            features['label'] = label
            features['video_id'] = prefix  # Use prefix as ID
            data.append(features)

df = pd.DataFrame(data)
df.to_csv(output_csv, index=False)
print(f'Saved features to {output_csv}')