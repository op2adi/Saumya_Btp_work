# Emotion Recognition from Speech and Video

## Overview
This project focuses on analyzing and processing audio-visual data from video files to perform emotion recognition. The workflow includes data loading, preprocessing, feature extraction, data augmentation, and model training using various machine learning and deep learning techniques.

---

## Dataset

The dataset used in this project comes from the **RAVDESS dataset**, which contains:

- **24 professional actors** (12 male, 12 female)
- **7 emotions**: Calm, Happy, Sad, Angry, Fearful, Disgust, Surprised + Neutral
- **2 statements** spoken in a neutral North American accent
- **2 levels** of emotional intensity

---

## Repository

GitHub: [Emotion Recognition Repository](https://github.com/op2adi/Saumya_Btp_work.git)

---

## Project Pipeline

### 1. **Data Loading and Preprocessing**
- Load video files and extract metadata.
- Convert audio to mono format.
- Remove silence from audio signals.
- Perform data augmentation to balance class distributions.

### 2. **Feature Extraction**
- Extract audio features such as:
  - **MFCCs**
  - **Chroma**
  - **Spectral Centroid**
  - and more.
- Normalize features using **StandardScaler**.

### 3. **Model Training and Evaluation**
- Train multiple models:
  - **Random Forest**
  - **Simple Neural Network**
  - **CNN**
  - **Complex CNN with BatchNorm**
- Evaluate models using the following metrics:
  - **Accuracy**
  - **F1 Score**
  - **ROC AUC**

---

## Results

| Model         | Accuracy | F1 Score | ROC AUC |
|---------------|----------|----------|---------|
| Random Forest | 0.692    | 0.673    | 0.947   |
| Neural Network| 0.737    | 0.735    | 0.954   |
| Simple CNN    | 0.715    | 0.714    | 0.942   |
| Complex CNN   | 0.762    | 0.763    | 0.960   |

### Key Findings:
- **Complex CNN with Batch Normalization** performed the best overall.
- **Random Forest** showed strong baseline performance.
- All models achieved **ROC AUC > 0.90**, indicating good classification capabilities.
- More complex architectures generally led to better performance.

---

## Dependencies

Ensure the following dependencies are installed:

```plaintext
- Python 3.7+
- librosa
- moviepy
- tensorflow
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
```

---

## Usage

1. Clone the repository:

```bash
git clone https://github.com/op2adi/Saumya_Btp_work.git
```

2. Install the required packages:

```bash
pip install -r requirements.txt
```

3. Run the Jupyter Notebook:

```bash
jupyter notebook kaam_start.ipynb
```

---

## Future Work

- Experiment with more advanced architectures like **transformers**.
- Add **video feature analysis** using computer vision.
- Deploy the model as a **web application**.
- Explore **real-time inference** capabilities.

---

## Contributors

- **Aditya Upadhyay**
- Under the guidance of **Dr. Saumya Yadav GOD**
