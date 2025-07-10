# 🫀 Heart Sound Classification using Deep Learning

## 📌 Overview

This project explores the automatic classification of heart sound recordings (phonocardiograms) into **normal** and **abnormal** categories using deep learning. It was developed as part of my final year BSc (Hons) Computer Science dissertation, and achieved promising results using both a custom CNN model and pretrained audio models.

The goal is to assist in early detection of cardiovascular abnormalities by leveraging machine learning techniques.

---

## 📊 Dataset

- **Source**: PhysioNet Challenge 2021 – [https://physionet.org/content/challenge-2016/1.0.0/](https://physionet.org/content/challenge-2016/1.0.0/)
- **Content**: Heart sound recordings in WAV format, labeled as normal or abnormal
- 🛑 **Note**: The dataset is not included in this repository due to licensing. Please download it directly from the PhysioNet website and place it in the `data/` folder.

---

## 🧠 Project Structure

### 🔹 Custom CNN Model
- Preprocessing using **mel-spectrograms** and **decibel scaling**
- Stratified K-Fold cross-validation (prioritizing abnormal F1-score)
- Techniques used: dropout, batch normalization, early stopping, class weighting
- Evaluation via confusion matrix, classification report, and F1-score

### 🔹 Pretrained Models
- Transfer learning using:
  - InceptionV3
  - VGGish
  - YAMNet
- Applied early stopping, learning rate reduction, and stratified folds
- Compared results across models using average performance metrics

---

## 📈 Sample Visuals

| CNN Confusion Matrix | Mel-Spectrogram |
|----------------------|-----------------|
| ![Confusion Matrix](plots/cnn_confusion_matrix.png) | ![Spectrogram](plots/example_spectrogram.png) |

> 📝 *If plots are not visible, ensure images are in the `plots/` folder with matching filenames.*

---

## 🛠️ Tools & Libraries

- Python
- TensorFlow / Keras
- NumPy / Pandas
- Librosa
- Matplotlib / Seaborn
- Scikit-learn

---

## 🚀 How to Run

1. Clone the repository:
```bash
git clone https://github.com/Crackedizzy/heart-sound-classification.git
cd heart-sound-classification
```

2. Download the dataset from PhysioNet and place files in the data/ folder.

3. Open and run:

    - notebooks/custom_cnn_model.ipynb

    - notebooks/pretrained_models.ipynb

---
## 📌 Results Summary

| Model        | F1 Score (Abnormal) | Accuracy | Recall (Abnormal) | Precision (Abnormal) |
|--------------|---------------------|----------|--------|-----------|
| Custom CNN   |       0.74          |   88%    |  88%   | 64%     |
| VGGish       |       0.78          |   92%    | 75%  | 82%     |
| InceptionV3  |       0.74          |   90%    | 70%  | 79%     |
| YAMNet       |       0.78          |   92%    | 73%  | 84%     |

---
## 💡 Future Improvements

- Improve class balance with data augmentation
- Explore attention-based audio models
- Build a Streamlit demo for audio upload + live classification

---
## 👨‍💻 Author

**Israel Morakinyo**  
BSc (Hons) Computer Science – First Class  
[GitHub](https://github.com/Crackedizzy) | [LinkedIn](https://www.linkedin.com/in/israel-morakinyo-98b00a204/)
