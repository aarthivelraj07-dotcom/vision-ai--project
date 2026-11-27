# Vision AI – Multi-Task Computer Vision Application

A Python-based Vision AI application that brings multiple computer vision and machine learning tasks together in one interactive dashboard.

The project uses machine learning and computer vision models to analyze images and support real-time or image-based recognition tasks.

## Features

The application includes six main tasks:

1. **Attendance & Emotion Detection** – Face recognition-based attendance support and emotion detection.
2. **Animal Detection** – Detects supported animal categories in images.
3. **Drowsiness Detection** – Analyzes facial and eye features to identify signs of drowsiness.
4. **Profile Analysis** – Performs facial profile analysis, including age estimation.
5. **Indian Sign Language (ISL) Recognition** – Recognizes supported static sign-language classes.
6. **Traffic Analysis** – Performs vehicle-related analysis, including vehicle detection and color analysis.

## Technologies Used

- Python
- Streamlit
- OpenCV
- TensorFlow / Keras
- Scikit-learn
- NumPy
- Pandas
- ONNX Runtime / ONNX models
- Joblib

## Project Structure

```text
Vision-AI/
├── app.py
├── models/
│   ├── attendance/
│   ├── animal_detection/
│   ├── drowsiness/
│   └── sign_language/
├── modules/
├── scripts/
├── tests/
├── training/
├── requirements.txt
└── README.md
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/aarthivelraj07-dotcom/vision-ai--project.git
cd vision-ai--project
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

**Windows**

```bash
.venv\Scripts\activate
```

**Linux / macOS**

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Run the Application

Start the Streamlit application:

```bash
python -m streamlit run app.py
```

Open the local URL displayed in the terminal, usually:

```text
http://localhost:8501
```

## How to Use

1. Launch the application.
2. Select the required task from the dashboard.
3. Provide the supported image, video, or other input for that task.
4. Run the detection or analysis.
5. Review the output displayed in the application.

Some features may require registration, configuration, or additional input before they can be used.

## Models and Datasets

The application uses pretrained and trained machine learning models for its supported tasks.

Model files and datasets may have separate licensing, attribution, privacy, or usage requirements. Please review the relevant licenses and acknowledgements before redistributing them.

Datasets containing biometric information, face images, attendance records, or personal data should not be shared publicly without appropriate permission.

## Disclaimer

This project is developed for educational and demonstration purposes. Model predictions may be inaccurate and should not be treated as professional, medical, identity, or security decisions.

## Author

Developed as an academic Vision AI project.
