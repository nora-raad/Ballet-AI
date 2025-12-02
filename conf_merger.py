import json
import math
import numpy as np
import os

json_dir = r'./training_data_json2'  # Adjust
BODY_25_FOOT_INDICES = range(19, 25)  # Foot points: big/small toes, heels L/R

def get_point(kps, idx, min_conf=0.3):
    x, y, c = kps[idx*3:idx*3+3]
    return (x, y, c) if c > min_conf else None

for json_file in sorted(os.listdir(json_dir)):
    if not json_file.endswith('.json'): continue
    with open(os.path.join(json_dir, json_file), 'r') as f:
        data = json.load(f)
    if not data['people']: continue

    # Merge keypoints: 25 points, take max conf
    merged_kps = [0] * 75  # Flat array
    for person in data['people']:
        kps = person['pose_keypoints_2d']
        for i in range(75):
            if kps[i] > merged_kps[i]:  # Overwrite if higher conf (for c values)
                merged_kps[i] = kps[i]

    # Now analyze merged

    lknee = get_point(merged_kps, 13, 0.5)
    lankle = get_point(merged_kps, 14, 0.5)
    rknee = get_point(merged_kps, 10, 0.5)
    if not (lknee and lankle and rknee):
        print(f"Frame {json_file}: Invalid - missing key legs")
        continue

    foot_points = [get_point(merged_kps, idx, 0.3) for idx in BODY_25_FOOT_INDICES if get_point(merged_kps, idx, 0.3)]
    if len(foot_points) < 4:
        print(f"Frame {json_file}: Invalid - too few feet points")
        continue
