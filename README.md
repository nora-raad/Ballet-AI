# Athletic Position Detection System
# سیستم تشخیص خودکار پوزیشن‌های پایه ورزشی

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![OpenPose](https://img.shields.io/badge/OpenPose-1.7.0-green.svg)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10+-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

*An AI-powered system for automatic detection and classification of five basic athletic positions using computer vision and machine learning*



</div>


## 🎯 Overview

This project implements an intelligent system for detecting and classifying five basic athletic positions using pose estimation and machine learning. The system provides real-time feedback through an intuitive web interface with Persian language support.

### Key Highlights

- **Dual Pipeline Architecture**: OpenPose for accuracy, MediaPipe for real-time performance
- **71% Test Accuracy**: Robust RandomForest classifier with comprehensive feature engineering
- **Real-time Detection**: Instant feedback within 2 seconds
- **User-Friendly Interface**: Clean Streamlit web app with Persian RTL support
- **Privacy-First**: All processing done locally, no data transmission

---

## ✨ Features

### Core Capabilities

- ✅ **Multi-Input Support**
  - Upload pre-recorded videos (MP4, AVI, MOV)
  - Live camera feed for real-time detection
  
- ✅ **Accurate Position Detection**
  - Five position classification (First, Second, Third, Fourth, Fifth)
  - Confidence scoring for predictions
  - Visual feedback with position diagrams

- ✅ **Intelligent Processing**
  - Automatic keypoint extraction
  - ~30 geometric features computed
  - Scale-independent normalization
  - Rule-based refinement for challenging positions

- ✅ **User Experience**
  - Persian language interface (RTL layout)
  - Helpful tips for each position
  - Error handling with clear messages
  - Responsive design

---

## 🎬 Demo

### Video Upload Mode
```
1. Upload video file
2. OpenPose extracts keypoints
3. System predicts position
4. See results with confidence score
```

### Live Camera Mode
```
1. Enable camera access
2. MediaPipe tracks in real-time
3. Hold position for 3 seconds
4. Get instant feedback
```

---

## 🏗️ Architecture

### System Overview

```
┌─────────────────────────────────────────────────────┐
│                  INPUT LAYER                        │
│  📹 Video Upload          📷 Live Camera            │
└────────────┬──────────────────────┬─────────────────┘
             │                      │
             ▼                      ▼
    ┌────────────────┐    ┌────────────────┐
    │   OpenPose     │    │   MediaPipe    │
    │   BODY_25      │    │   33 Points    │
    │   25 keypoints │    │   Real-time    │
    └────────┬───────┘    └────────┬───────┘
             │                      │
             └──────────┬───────────┘
                        │
                        ▼
            ┌──────────────────────┐
            │  Feature Extraction  │
            │  utils.py            │
            │  ~30 features        │
            └──────────┬───────────┘
                       │
                       ▼
            ┌──────────────────────┐
            │  StandardScaler      │
            │  Normalization       │
            └──────────┬───────────┘
                       │
                       ▼
            ┌──────────────────────┐
            │  RandomForest        │
            │  Classifier          │
            └──────────┬───────────┘
                       │
                       ▼
            ┌──────────────────────┐
            │  Refinement          │
            │  refine_4th_5th()    │
            └──────────┬───────────┘
                       │
                       ▼
            ┌──────────────────────┐
            │  Web Interface       │
            │  Streamlit UI        │
            └──────────────────────┘
```

### Data Flow

```
Raw Input → Pose Estimation → Feature Extraction → 
Scaling → Classification → Refinement → Output
```

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- OpenPose 1.7.0 (for video processing)
- Webcam (for live detection)
- Windows 10/11 (or Linux with modifications)

### Step 1: Clone Repository

```bash
git clone https://github.com/nora-raad/Ballet-AI.git
cd Ballet-AI
```

### Step 2: Install Python Dependencies

```bash
pip install -r requirements.txt
```

**requirements.txt:**
```txt
streamlit==1.28.0
opencv-python==4.8.1
numpy==1.24.3
pandas==2.0.3
scikit-learn==1.3.0
joblib==1.3.2
mediapipe==0.10.5
streamlit-webrtc==0.47.1
Pillow==10.0.0
av==10.0.0
```

### Step 3: Install OpenPose

1. Download OpenPose 1.7.0 binaries from [official repository](https://github.com/CMU-Perceptual-Computing-Lab/openpose/releases)
2. Extract to a directory (e.g., `C:\openpose`)
3. Update the path in `app.py` line 191:
   ```python
   openpose_bin = r'YOUR_PATH\openpose\bin\OpenPoseDemo.exe'
   ```

### Step 4: Prepare Assets

Ensure you have the following structure:
```
project/
├── images/
│   ├── header.png
│   ├── first.png
│   ├── second.png
│   ├── third.png
│   ├── fourth.png
│   └── fifth.png
├── ballet_rf_model1.pkl
├── ballet_rf_scaler1.pkl
└── app.py
```

### Step 5: Run the Application

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

---

## 💻 Usage

### Quick Start

1. **Launch Application**
   ```bash
   streamlit run app.py
   ```

2. **Choose Input Mode**
   - Select "بارگذاری ویدیو" (Upload Video) for pre-recorded videos
   - Select "دوربین زنده" (Live Camera) for real-time detection

3. **Get Results**
   - View detected position name
   - Check confidence percentage
   - Read helpful tips

### Video Upload

```python
# Supported formats: MP4, AVI, MOV
# Recommended: 640x480 or higher, 30 fps
# Content: Full body visible, especially legs and feet
```

### Live Camera

```python
# Requirements:
# - Good lighting
# - Full body in frame
# - Hold position for 3+ seconds
# - Stable posture
```

---

## 📊 Dataset

### Data Collection

- **Source**: Video recordings of individuals performing 5 positions
- **Total Samples**: ~1,000-1,500 frames
- **Classes**: 5 balanced classes (0-4)
- **Labeling**: Manual annotation based on position type

### Position Definitions

| Class | Name | Description |
|-------|------|-------------|
| 0 | First | Heels together, toes out in straight line |
| 1 | Second | Feet shoulder-width apart, toes out |
| 2 | Third | One foot front, heel touches middle of back foot |
| 3 | Fourth | Feet separated with visible gap (one foot length) |
| 4 | Fifth | Feet crossed, heel touches toes |

### Data Preprocessing

```python
# OpenPose Pipeline
Videos → OpenPose → JSON keypoints → 
conf_merger.py → preprocess1.py → CSV dataset

# MediaPipe Pipeline
Live captures → MediaPipe landmarks → 
preprocess2.py → CSV dataset
```

---

## 📈 Model Performance

### Overall Metrics

| Metric | Validation | Test |
|--------|-----------|------|
| **Accuracy** | 89.2% | 71.0% |
| **Precision** | 0.88 | 0.73 |
| **Recall** | 0.89 | 0.71 |
| **F1-Score** | 0.88 | 0.70 |

### Per-Class Performance

| Position | Precision | Recall | F1-Score | Support |
|----------|-----------|--------|----------|---------|
| First (0) | 1.00 | 1.00 | 1.00 | 5 |
| Second (1) | 1.00 | 1.00 | 1.00 | 4 |
| Third (2) | 0.83 | 0.83 | 0.83 | 6 |
| Fourth (3) | 0.44 | 0.44 | 0.44 | 9 |
| Fifth (4) | 0.57 | 0.57 | 0.57 | 7 |

### Model Comparison

| Model | Validation Acc | Test Acc | Decision |
|-------|----------------|----------|----------|
| **RandomForest** | 89% | 71% | ✅ Selected |
| SVM (RBF) | 89% | 32% | ❌ Rejected (Overfitting) |

### Feature Importance (Top 10)

```
1. ankle_x_dist           15.2%
2. ankle_dist             14.1%
3. foot_spread            11.3%
4. back_heel_to_front_*    9.8%
5. cross_factor            8.5%
6. foot_y_std              6.7%
7. left_turnout_angle      5.9%
8. right_turnout_angle     5.4%
9. heel_toe_overlap_*      4.8%
10. left_leg_len           4.2%
```

---

## 📁 Project Structure

```
Ballet-AI/
│
├── app.py                          # Main Streamlit application
├── utils.py                        # Feature extraction utilities
├── conf_merger.py                  # Fix OpenPose multi-person detection
│
├── preprocess1.py                  # OpenPose data preprocessing
├── preprocess2.py                  # MediaPipe data preprocessing
│
├── train_random_forest.py          # RF model training
├── train_svm.py                    # SVM model training (comparison)
│
├── ballet_rf_model1.pkl            # Trained RandomForest (OpenPose)
├── ballet_rf_model2.pkl            # Trained RandomForest (MediaPipe)
├── ballet_rf_scaler1.pkl           # StandardScaler (OpenPose)
├── ballet_rf_scaler2.pkl           # StandardScaler (MediaPipe)
│
├── ballet_svm_model1.pkl           # Trained SVM (comparison)
├── ballet_scaler1.pkl              # SVM scaler
│
├── images/                         # UI assets
│   ├── header.png
│   ├── first.png
│   ├── second.png
│   ├── third.png
│   ├── fourth.png
│   └── fifth.png
│
├── data/                           # Temporary processing folder
│   └── (created at runtime)
│
├── requirements.txt                # Python dependencies
├── README.md                       # This file
└── LICENSE                         # MIT License
```

---

## 🛠️ Technologies

### Computer Vision

- **OpenPose 1.7.0**: High-accuracy pose estimation (25 keypoints)
- **MediaPipe 0.10**: Real-time pose tracking (33 landmarks)
- **OpenCV 4.8**: Image processing and video handling

### Machine Learning

- **scikit-learn 1.3**: RandomForest classifier, StandardScaler
- **NumPy 1.24**: Numerical computations
- **Pandas 2.0**: Data manipulation

### Web Framework

- **Streamlit 1.28**: Web interface
- **streamlit-webrtc 0.47**: Real-time video streaming

### Development

- **Python 3.8+**: Core programming language
- **joblib**: Model serialization

---

## 🔧 Challenges & Solutions

### Challenge 1: Multi-Person Detection

**Problem**: OpenPose sometimes split one person into multiple detections

**Solution**: 
```python
# conf_merger.py
- Analyze spatial proximity between detections
- Merge keypoints belonging to same person
- Keep highest confidence scores
```

### Challenge 2: Fourth ↔ Fifth Confusion

**Problem**: 50% error rate between positions 4 and 5 (very similar in 2D)

**Solution**:
```python
# refine_fourth_fifth() function
- Measure back_heel_to_front_bigtoe distance
- If < 0.08 → Fifth (heels touching)
- If > 0.12 → Fourth (clear separation)
- Hybrid ML + rule-based approach
```

### Challenge 3: SVM Overfitting

**Problem**: SVM showed severe overfitting (89% → 32%)

**Solution**:
```python
# Selected RandomForest instead
- More stable: 89% → 71%
- Better generalization
- Interpretable feature importance
```

### Challenge 4: Real-time Stability

**Problem**: False detections when moving between positions

**Solution**:
```python
# Implemented stability checks
- Require 3 consecutive valid frames
- Fast reset if keypoints lost (0.3s interval)
- Visibility threshold (0.5)
```

---

## 🚀 Future Work

### Short-term Improvements

- [ ] Increase training dataset size (>2000 samples)
- [ ] Add more athletic positions (>5)
- [ ] Implement cross-validation
- [ ] Fine-tune hyperparameters

### Medium-term Enhancements

- [ ] 3D pose estimation (depth camera support)
- [ ] Multi-person detection support
- [ ] Mobile application (Android/iOS)
- [ ] User feedback mechanism

### Long-term Vision

- [ ] Deep learning models (CNN/LSTM)
- [ ] Ensemble methods (RF + SVM + NN)
- [ ] Video sequence analysis (temporal features)
- [ ] Cloud deployment option
- [ ] Integration with sports training apps

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. **Fork the repository**
   ```bash
   git fork https://github.com/nora-raad/Ballet-AI.git
   ```

2. **Create a feature branch**
   ```bash
   git checkout -b feature/AmazingFeature
   ```

3. **Commit changes**
   ```bash
   git commit -m 'Add some AmazingFeature'
   ```

4. **Push to branch**
   ```bash
   git push origin feature/AmazingFeature
   ```

5. **Open a Pull Request**

### Contribution Guidelines

- Follow PEP 8 style guide
- Add docstrings to functions
- Include unit tests
- Update documentation
- Comment your code


---

## 🙏 Acknowledgments

### Research & Tools

- **OpenPose**: [CMU Perceptual Computing Lab](https://github.com/CMU-Perceptual-Computing-Lab/openpose)
  - Cao et al., "OpenPose: Realtime Multi-Person 2D Pose Estimation" (2019)
  
- **MediaPipe**: [Google Research](https://google.github.io/mediapipe/)
  - Bazarevsky et al., "BlazePose: On-device Real-time Body Pose Tracking" (2020)
  
- **RandomForest**: Breiman, L., "Random Forests" (2001)
  
- **scikit-learn**: Pedregosa et al., "Scikit-learn: Machine Learning in Python" (2011)

### Inspiration

- Athletic training and position correction systems
- Computer vision applications in sports
- Pose estimation research community

### Special Thanks

- Science and Research Branch, Islamic Azad University, Tehran for their Academic support
- My Thesis advisor DR. Farsad Zamani Boroujeni and committee members
- https://www.youtube.com/@NicholasRenotte for inspiring me to take on this challenging project.

---

## 📧 Contact

**Nora Raad**
- GitHub: [@nora-raad](https://github.com/nora-raad)
- Email: [nooraraad@gmail.com]
- LinkedIn: [https://www.linkedin.com/in/nora-raad]
- Project Link: [https://github.com/nora-raad/Ballet-AI](https://github.com/nora-raad/Ballet-AI)

---

## 📊 Project Stats

![GitHub stars](https://img.shields.io/github/stars/nora-raad/Ballet-AI?style=social)
![GitHub forks](https://img.shields.io/github/forks/nora-raad/Ballet-AI?style=social)
![GitHub issues](https://img.shields.io/github/issues/nora-raad/Ballet-AI)
![GitHub pull requests](https://img.shields.io/github/issues-pr/nora-raad/Ballet-AI)

---

## 🔖 Citation

If you use this project in your research, please cite:

```bibtex
@misc{raad2024athletic,
  author = {Raad, Nora},
  title = {Athletic Position Detection System using Pose Estimation and Machine Learning},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/nora-raad/Ballet-AI}}
}
```

---

<div align="center">

**Made with ❤️ by [Nora Raad](https://github.com/nora-raad)**

⭐ Star this repo if you find it helpful!

[⬆ Back to Top](#athletic-position-detection-system)

</div>
