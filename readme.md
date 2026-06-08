# 🔍 Face Recognition System (FaceNet + MTCNN)

A real-time AI-based Face Recognition system built using **FaceNet embeddings** and **MTCNN face detection**, deployed with a **Streamlit web interface**.

This project supports image upload recognition, webcam-based detection (local only), and model performance visualization.

---

# 🚀 Features

### 📸 Face Recognition
- Upload an image and detect faces
- Recognize known persons using FaceNet embeddings
- Displays confidence score for each prediction

### 🎥 Webcam Recognition (Local Only)
- Real-time face detection using webcam
- Live prediction overlay on video frames

### 📊 Model Evaluation
- Accuracy curve over training epochs
- Loss curve visualization
- Precision and Recall metrics

### 🧠 AI Model
- FaceNet for generating embeddings (`embedding.py`)
- MTCNN for face detection (`detection.py`)
- Cosine / distance-based similarity matching

---

# 🏗️ Project Structure

FaceDetectionRecognition/

├── app.py                  # Streamlit UI
├── requirements.txt        # Dependencies
├── README.md               # Documentation
│
├── src/
│   ├── recognition.py      # Main recognition pipeline
│   ├── embedding.py        # Face embedding generation (FaceNet)
│   ├── detection.py        # Face detection (MTCNN)
│
├── models/
│   ├── embeddings.pkl      # Stored face embeddings
│
├── dataset/ 
│   ├── train/
│   ├── test/               # Test images


---

# ⚙️ Installation & Setup

## 1️⃣ Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/FaceRecognitionSystem.git
cd FaceRecognitionSystem