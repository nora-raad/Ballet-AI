import streamlit as st
import cv2
import numpy as np
import subprocess
import json
import os
import tempfile
import time
import math
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase
import av
from utils import extract_features_from_video, compute_angle, BODY_25_INDICES, FOOT_INDICES
import joblib
import pandas as pd
import threading
import mediapipe as mp
# from sklearn.metrics import accuracy_score, classification_report
from PIL import Image

# -----------------------------------------------------------
# تنظیمات صفحه
# -----------------------------------------------------------
st.set_page_config(page_title="سیستم تشخیص حرکت", layout="wide")

# -----------------------------------------------------------
# استایل CSS
# -----------------------------------------------------------
st.markdown("""
<style>
.stApp {
    background-color: #D5FFFF;
}

.rtl {
    direction: rtl;
    text-align: right;
    font-family: 'Vazir', 'Tahoma', 'Arial', sans-serif;
}
            
div[role="radiogroup"] {
    direction: rtl;
    justify-content: flex-end;
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------
# تصویر هدر
# -----------------------------------------------------------
header_image = Image.open("images/header.png")
col1, col2, col3 = st.columns([0.75, 2, 0.75])
with col2:
    st.image(header_image, use_container_width=True)

# -----------------------------------------------------------
# تنظیمات اولیه
# -----------------------------------------------------------
DATA_DIR = './data'
os.makedirs(DATA_DIR, exist_ok=True)

clf = joblib.load('ballet_rf_model1.pkl')
scaler = joblib.load('ballet_rf_scaler1.pkl')

# -----------------------------------------------------------
# اطلاعات پوزیشن‌ها
# -----------------------------------------------------------
position_instructions = {
    0: {
        'name_fa': 'پوزیشن اول',
        'tips_fa': 'پاشنه‌ها کنار هم، انگشتان پا به بیرون چرخیده و یک خط مستقیم تشکیل می‌دهند. زانوها صاف، وضعیت بدن راست.',
        'image': './images/first.png'
    },
    1: {
        'name_fa': 'پوزیشن دوم',
        'tips_fa': 'پاها به اندازه عرض شانه از هم فاصله دارند، انگشتان پا به بیرون. زانوها صاف، وزن بدن یکنواخت.',
        'image': './images/second.png'
    },
    2: {
        'name_fa': 'پوزیشن سوم',
        'tips_fa': 'یک پا جلوی دیگری، پاشنه پای جلو وسط پای پشت را لمس می‌کند. انگشتان پا به بیرون چرخیده.',
        'image': './images/third.png'
    },
    3: {
        'name_fa': 'پوزیشن چهارم',
        'tips_fa': 'یک پا جلوی دیگری با فاصله (حدود یک طول پا). انگشتان پا به بیرون. فاصله قابل مشاهده بین مچ پاها.',
        'image': './images/fourth.png'
    },
    4: {
        'name_fa': 'پوزیشن پنجم',
        'tips_fa': 'پاها متقاطع، پاشنه پای جلو انگشت پای پشت را لمس می‌کند. انگشتان پا کاملاً به بیرون چرخیده.',
        'image': './images/fifth.png'
    }
}

# -----------------------------------------------------------
# تابع نمایش اطلاعات پوزیشن
# -----------------------------------------------------------
def display_position_info(prediction, confidence):
    """نمایش اطلاعات پوزیشن با فارسی و RTL"""
    info = position_instructions.get(prediction, {
        'name_fa': 'نامشخص',
        'tips_fa': 'توضیحی موجود نیست.',
        'image': None
    })
    
    # نمایش درصد اطمینان
    st.markdown(f"""
        <div style='direction: rtl; text-align: right;'>
            <p style='font-size: 1.2rem; font-weight: bold;'>درجه اطمینان: {confidence:.1f}%</p>
        </div>
    """, unsafe_allow_html=True)
    
    # ستون‌ها برای تصویر و متن
    col_img, col_text = st.columns([1, 2])
    
    with col_img:
        if info.get('image') and os.path.exists(info['image']):
            st.image(info['image'], use_container_width=True)
    
    with col_text:
        st.markdown(f"""
            <div style="direction: rtl; text-align: right;">
                <div style="background-color: #d4edda; padding: 1.2rem; border-radius: 0.5rem; margin-bottom: 1rem; border-right: 4px solid #28a745;">
                    <h2 style="margin: 0; color: #155724; font-size: 1.8rem;">{info['name_fa']}</h2>
                </div>
                <div style="background-color: #d1ecf1; padding: 1.2rem; border-radius: 0.5rem; border-right: 4px solid #17a2b8;">
                    <p style="margin: 0; line-height: 1.9; font-size: 1.1rem;">💡 {info['tips_fa']}</p>
                </div>
            </div>
        """, unsafe_allow_html=True)

# -----------------------------------------------------------
# توابع پیش‌بینی
# -----------------------------------------------------------
def predict_pose(json_files):
    features = extract_features_from_video(json_files)
    if features is None:
        return None, None
    df = pd.DataFrame([features])
    scaled = scaler.transform(df)
    pred = clf.predict(scaled)[0]
    probas = clf.predict_proba(scaled)[0]
    pred = refine_fourth_fifth(pred, features, probas)
    confidence = max(probas) * 100
    return pred, confidence

def refine_fourth_fifth(prediction, features, probas):
    """تفکیک بهتر بین پوزیشن چهارم و پنجم"""
    if prediction not in [3, 4]:
        return prediction
    
    back_heel_to_front_bigtoe = features.get('back_heel_to_front_bigtoe', 0)
    back_heel_to_front_smalltoe = features.get('back_heel_to_front_smalltoe', 0)
    cross_factor = features.get('cross_factor', 0)
    avg_heel_toe_dist = (back_heel_to_front_bigtoe + back_heel_to_front_smalltoe) / 2
    
    FOURTH_MIN_DEPTH = 0.10
    FIFTH_MAX_DEPTH = 0.20
    
    if abs(cross_factor) != 1:
        return prediction
    
    if avg_heel_toe_dist > FOURTH_MIN_DEPTH and prediction == 4:
        if probas[3] > 0.05:
            print(f"✓ Override: Fifth→Fourth (depth={avg_heel_toe_dist:.3f})")
            return 3
    elif avg_heel_toe_dist < FIFTH_MAX_DEPTH and prediction == 3:
        if probas[4] > 0.05:
            print(f"✓ Override: Fourth→Fifth (depth={avg_heel_toe_dist:.3f})")
            return 4
    
    return prediction

# -----------------------------------------------------------
# اجرای OpenPose
# -----------------------------------------------------------
def run_openpose(input_path, output_dir):
    import shlex
    openpose_bin = r'C:\Users\noora\Downloads\openpose-1.7.0-binaries-win64-gpu-python3.7-flir-3d_recommended\openpose\bin\OpenPoseDemo.exe'
    openpose_root = os.path.dirname(os.path.dirname(openpose_bin))
    model_folder = os.path.join(openpose_root, 'models')

    input_path_abs = os.path.abspath(input_path)
    output_dir_abs = os.path.abspath(output_dir)
    os.makedirs(output_dir_abs, exist_ok=True)

    images_output_abs = os.path.join(output_dir_abs, 'images')
    os.makedirs(images_output_abs, exist_ok=True)

    cmd = [
        openpose_bin,
        '--video', input_path_abs,
        '--write_json', output_dir_abs,
        '--model_folder', model_folder,
        '--model_pose', 'BODY_25',
        '--display', '0',
        '--write_images', images_output_abs,
        '--write_images_format', 'png',
        '--render_pose', '2',
        '--net_resolution', '-1x320',
        '--disable_blending'
    ]

    cmd_str = " ".join(shlex.quote(p) for p in cmd)
    st.markdown("""
        <div style="direction: rtl; text-align: right; background-color: #e3f2fd; padding: 1rem; border-radius: 0.5rem; border-right: 4px solid #2196f3;">
            در حال پردازش با OpenPose...
        </div>
    """, unsafe_allow_html=True)
    print(f"Running OpenPose: {cmd_str}")

    try:
        result = subprocess.run(
            cmd, cwd=openpose_root, check=True, 
            capture_output=True, text=True, timeout=300
        )
        print("OpenPose stdout:", result.stdout)
    except subprocess.CalledProcessError as e:
        st.error(f"خطا در OpenPose - کد خطا: {e.returncode}")
        print("OpenPose error:", e.stderr)
        return []
    except Exception as e:
        st.error(f"خطا در اجرای OpenPose: {e}")
        return []

    json_files = [
        os.path.join(output_dir_abs, f) 
        for f in os.listdir(output_dir_abs) 
        if f.endswith('_keypoints.json')
    ]
    json_files.sort()
    return json_files

# -----------------------------------------------------------
# تنظیمات MediaPipe
# -----------------------------------------------------------
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

MP_INDICES = {
    'LHip': 23, 'RHip': 24,
    'LKnee': 25, 'RKnee': 26,
    'LAnkle': 27, 'RAnkle': 28,
    'LHeel': 29, 'RHeel': 30,
    'LBigToe': 31, 'RBigToe': 32,
    'LSmallToe': 31, 'RSmallToe': 32
}

def compute_angle_mediapipe(p1, p2, p3):
    if p1 is None or p2 is None or p3 is None:
        return 0
    v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    return np.degrees(angle)

def extract_features_from_mediapipe(img):
    with mp_pose.Pose(min_detection_confidence=0.3, min_tracking_confidence=0.3) as pose:
        results = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if not results.pose_landmarks:
            return None
        
        landmarks = results.pose_landmarks.landmark
        points = {}
        
        for name, idx in MP_INDICES.items():
            lm = landmarks[idx]
            if lm.visibility > 0.2:
                points[name] = (lm.x * img.shape[1], lm.y * img.shape[0], lm.visibility)
            else:
                points[name] = None
        
        required_points = ['LKnee', 'LAnkle', 'RKnee', 'RAnkle', 'LHip', 'RHip']
        if any(points[name] is None for name in required_points):
            return None
        
        foot_points = [points[name] for name in ['LBigToe', 'LSmallToe', 'LHeel', 'RBigToe', 'RSmallToe', 'RHeel'] if points[name]]
        if len(foot_points) < 3:
            return None
        
        lankle, rankle = points['LAnkle'], points['RAnkle']
        lknee, rknee = points['LKnee'], points['RKnee']

        left_leg_len = math.dist(lknee[:2], lankle[:2])
        right_leg_len = math.dist(rknee[:2], rankle[:2])
        avg_leg_len = (left_leg_len + right_leg_len) / 2

        if avg_leg_len < 1:
            return None

        if lankle[0] > rankle[0]:
            cross_factor = 1
        elif rankle[0] > lankle[0]:
            cross_factor = -1
        else:
            cross_factor = 0

        left_foot_vis = sum(1 for name in ['LBigToe', 'LSmallToe', 'LHeel'] if points[name] is not None)
        right_foot_vis = sum(1 for name in ['RBigToe', 'RSmallToe', 'RHeel'] if points[name] is not None)

        if cross_factor == 1:
            back_heel_to_front_bigtoe = math.dist(points['RHeel'][:2], points['LBigToe'][:2]) / avg_leg_len if points['RHeel'] and points['LBigToe'] else 0
            back_heel_to_front_smalltoe = math.dist(points['RHeel'][:2], points['LSmallToe'][:2]) / avg_leg_len if points['RHeel'] and points['LSmallToe'] else 0
        elif cross_factor == -1:
            back_heel_to_front_bigtoe = math.dist(points['LHeel'][:2], points['RBigToe'][:2]) / avg_leg_len if points['LHeel'] and points['RBigToe'] else 0
            back_heel_to_front_smalltoe = math.dist(points['LHeel'][:2], points['RSmallToe'][:2]) / avg_leg_len if points['LHeel'] and points['RSmallToe'] else 0
        else:
            back_heel_to_front_bigtoe = 0
            back_heel_to_front_smalltoe = 0

        features = {
            'ankle_dist': math.dist(lankle[:2], rankle[:2]) / avg_leg_len,
            'foot_spread': (max(p[0] for p in foot_points) - min(p[0] for p in foot_points)) / avg_leg_len,
            'foot_y_std': np.std([p[1] for p in foot_points]) if foot_points else 0,
            'left_straightness': abs(lknee[0] - lankle[0]),
            'right_straightness': abs(rknee[0] - rankle[0]),
            'left_turnout_angle': compute_angle_mediapipe(points['LHip'], lankle, points['LBigToe']),
            'right_turnout_angle': compute_angle_mediapipe(points['RHip'], rankle, points['RBigToe']),
            'cross_factor': cross_factor,
            'left_leg_len': left_leg_len,
            'right_leg_len': right_leg_len,
            'heel_toe_overlap_left': math.dist(points['LHeel'][:2], points['RBigToe'][:2]) / avg_leg_len if points['LHeel'] and points['RBigToe'] else 0,
            'heel_toe_overlap_right': math.dist(points['RHeel'][:2], points['LSmallToe'][:2]) / avg_leg_len if points['RHeel'] and points['LSmallToe'] else 0,
            'ankle_x_dist': abs(lankle[0] - rankle[0]) / avg_leg_len,
            'left_foot_vis': left_foot_vis,
            'right_foot_vis': right_foot_vis,
            'back_heel_to_front_bigtoe': back_heel_to_front_bigtoe,
            'back_heel_to_front_smalltoe': back_heel_to_front_smalltoe,
        }
        return features

def predict_from_mediapipe(img):
    features = extract_features_from_mediapipe(img)
    if features is None:
        return None, None
    try:
        df = pd.DataFrame([features])
        scaled = scaler.transform(df)
        pred = clf.predict(scaled)[0]
        probas = clf.predict_proba(scaled)[0]
        pred = refine_fourth_fifth(pred, features, probas)
        confidence = max(probas) * 100
        return pred, confidence
    except Exception as e:
        print(f"Prediction error: {e}")
        return None, None

# -----------------------------------------------------------
# رابط کاربری
# -----------------------------------------------------------
st.markdown("<h1 style='direction: rtl; text-align: right;'>سیستم تشخیص پوزیشن</h1>", unsafe_allow_html=True)

st.markdown("""
    <div style='direction: rtl; text-align: right; background-color: #d1ecf1; padding: 1rem; border-radius: 0.5rem; margin: 1rem 0; border-right: 4px solid #17a2b8;'>
        <p style='margin: 0;'>⚠️ نکته: این سیستم از OpenPose برای ورودی ویدیو و از MediaPipe برای ورودی دوربین استفاده می‌کند.</p>
    </div>
""", unsafe_allow_html=True)

st.markdown("<p style='direction: rtl; text-align: right; font-weight: bold; margin-top: 1rem;'>روش ورودی را انتخاب کنید:</p>", unsafe_allow_html=True)


st.markdown("<div style='direction: rtl; text-align: right;'>", unsafe_allow_html=True)
input_mode = st.radio(
    'input_mode',
    ('دوربین زنده', 'بارگذاری ویدیو'),
    label_visibility='collapsed',
    horizontal=True
)
st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------------------------------------
# حالت دوربین
# -----------------------------------------------------------
if input_mode == 'دوربین زنده':
    st.markdown("<h2 style='direction: rtl; text-align: right;'>ورودی دوربین زنده</h2>", unsafe_allow_html=True)
    st.markdown("""
        <div style="direction: rtl; text-align: right; background-color: #fff3cd; padding: 1rem; border-radius: 0.5rem; margin: 1rem 0; border-right: 4px solid #ffc107;">
            <p style="margin-bottom: 0.5rem;">⚠️ حالت دوربین از MediaPipe استفاده می‌کند. برای دقت بیشتر، از حالت بارگذاری ویدیو استفاده کنید.</p>
            <p style="margin: 0;">اجازه دسترسی به دوربین را بدهید و هر پوزیشن را حداقل ۳ ثانیه نگه دارید.</p>
        </div>
    """, unsafe_allow_html=True)

    class VideoProcessor(VideoProcessorBase):
        def __init__(self):
            self.last_prediction_time = 0
            self.prediction = None
            self.confidence = None

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_ndarray(format="bgr24")
            current_time = time.time()
            
            if current_time - self.last_prediction_time >= 3.0:
                self.last_prediction_time = current_time
                pred, conf = predict_from_mediapipe(img)
                if pred is not None:
                    self.prediction = pred
                    self.confidence = conf
            
            try:
                with mp_pose.Pose(min_detection_confidence=0.3, min_tracking_confidence=0.3) as pose:
                    results = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    if results.pose_landmarks:
                        mp_drawing.draw_landmarks(
                            img, 
                            results.pose_landmarks, 
                            mp_pose.POSE_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                        )
            except Exception as e:
                print(f"Drawing error: {e}")
            
            return av.VideoFrame.from_ndarray(img, format="bgr24")

    col1, col2 = st.columns([2, 1])
    
    with col1:
        webrtc_ctx = webrtc_streamer(
            key="ballet-camera",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            video_processor_factory=VideoProcessor,
            async_processing=True,
            media_stream_constraints={"video": True, "audio": False}
        )
    
    with col2:
        result_placeholder = st.empty()
        
        if webrtc_ctx.video_processor:
            while webrtc_ctx.state.playing:
                if webrtc_ctx.video_processor.prediction is not None:
                    pred = webrtc_ctx.video_processor.prediction
                    conf = webrtc_ctx.video_processor.confidence
                    
                    with result_placeholder.container():
                        display_position_info(pred, conf)
                else:
                    with result_placeholder.container():
                        st.markdown("""
                            <div style="direction: rtl; text-align: right;">
                                <div style="background-color: #d1ecf1; padding: 1.5rem; border-radius: 0.5rem; border-right: 4px solid #17a2b8;">
                                    <p style="font-size: 1.2rem; margin-bottom: 0.5rem; font-weight: bold;">⏳ در انتظار تشخیص حرکت...</p>
                                    <p style="margin: 0;">لطفاً کل بدن خود را در کادر قرار دهید</p>
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
                
                time.sleep(0.5)

# -----------------------------------------------------------
# حالت بارگذاری ویدیو
# -----------------------------------------------------------
elif input_mode == 'بارگذاری ویدیو':
    st.markdown("<h2 style='direction: rtl; text-align: right;'>بارگذاری ویدیو</h2>", unsafe_allow_html=True)
    st.markdown("""
        <p style='direction: rtl; text-align: right; font-size: 1.1rem;'>
            فایل ویدیو خود را با یکی از فرمت‌های MP4، AVI یا MOV بارگذاری کنید
        </p>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader('', type=['mp4', 'avi', 'mov'], label_visibility='collapsed')

    if uploaded_file:
        video_save_path = os.path.join(DATA_DIR, uploaded_file.name)
        with open(video_save_path, 'wb') as f:
            f.write(uploaded_file.read())

        video_name_noext = os.path.splitext(uploaded_file.name)[0]
        output_dir = os.path.join(DATA_DIR, f'{video_name_noext}_openpose')
        os.makedirs(output_dir, exist_ok=True)

        with st.spinner('در حال پردازش با OpenPose...'):
            json_files = run_openpose(video_save_path, output_dir)

        if json_files:
            pred, confidence = predict_pose(json_files)
            if pred is not None:
                display_position_info(pred, confidence)
                
                # نمایش ویدیو کوچکتر
                st.markdown("<p style='direction: rtl; text-align: right; font-weight: bold; margin-top: 2rem;'>ویدیو بارگذاری شده:</p>", unsafe_allow_html=True)
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.video(video_save_path)
            else:
                st.markdown("""
                    <div style='direction: rtl; text-align: right; background-color: #f8d7da; padding: 1rem; border-radius: 0.5rem; border-right: 4px solid #dc3545;'>
                        <p style='margin: 0; color: #721c24;'>❌ حرکتی تشخیص داده نشد. لطفاً ویدیو دیگری امتحان کنید.</p>
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div style='direction: rtl; text-align: right; background-color: #f8d7da; padding: 1rem; border-radius: 0.5rem; border-right: 4px solid #dc3545;'>
                    <p style='margin: 0; color: #721c24;'>❌ پردازش ویدیو ناموفق بود. لطفاً دوباره تلاش کنید.</p>
                </div>
            """, unsafe_allow_html=True)