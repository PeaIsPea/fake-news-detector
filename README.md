
# 🧠 Fake News Detector Web App

This is a simple web application that classifies news articles as **Real** or **Fake** using a machine learning model trained on a news dataset. The app is built with **FastAPI** for the backend and a fun **pixel-art style** frontend using **HTML/CSS**.

## 🚀 Demo

Users can input a news **title** and **content**, then click the **Predict** button to check if the news is real or fake.

<p align="center">
  <img src="app/static/bg2.gif" width="600" alt="Demo Screenshot" />
</p>

---

## 📁 Project Structure

```
Tweet_Spam/
│
├── app/
│   ├── main.py                    # FastAPI app entry point
│   ├── static/                    # Static files (CSS, background gifs)
│   │   ├── styles.css
│   │   ├── bg.gif / bg2.gif / bg3.gif
│   ├── templates/
│   │   └── index.html             # HTML frontend
│
├── dataset/                       # Dataset files
│   ├── train.csv
│   ├── test.csv
│   └── valid.csv
│
├── model/
│   └── fake_news_classifier.pkl   # Trained RandomForest model
│
├── scripts/
│   ├── train_model.py             # Model training script
│   ├── Figure_1.png               # Confusion matrix
│   └── Figure_2.png               # Classification report
```

---

## 🧩 Model Performance

### Confusion Matrix

<p align="center">
  <img src="scripts/Figure_1.png" width="400" alt="Confusion Matrix">
</p>

- **True Positives (Real predicted as Real)**: 5685  
- **True Negatives (Fake predicted as Fake)**: 1622  
- **False Positives (Fake predicted as Real)**: 33  
- **False Negatives (Real predicted as Fake)**: 1038

---

### Classification Report

<p align="center">
  <img src="scripts/Figure_2.png" width="600" alt="Classification Report">
</p>

- **Accuracy**: 87%
- **F1-score (Fake)**: 0.75  
- **F1-score (Real)**: 0.91  
- The model performs especially well in identifying **real news**, and oversampling helps improve **recall** for fake news.

---

## 🛠️ Tech Stack

- **Python**
- **FastAPI**
- **Jinja2 Templates**
- **Pandas, Joblib, Scikit-learn**
- **HTML/CSS (pixel-art theme)**

---

## 📦 How to Run

1. Clone the repository:

```bash
git clone https://github.com/yourusername/fake-news-detector.git
cd fake-news-detector
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the FastAPI server:

```bash
uvicorn app.main:app --reload
```

4. Open your browser and go to `http://localhost:8000`

---

## ✨ Screenshots

### Web Interface

<p align="center">
  <img src="scripts/demo_ui.png" width="600" alt="UI Screenshot" />
</p>

---

## 🧪 Dataset

The dataset is split into:
- `train.csv`
- `test.csv`
- `valid.csv`

It includes labeled news with fields like `title`, `text`, and `label`.

---

## 🤖 Model Details

- **Algorithm**: Random Forest Classifier
- **Feature Selection**: Top 10% features
- **Criterion**: Gini impurity
- **Preprocessing**: TF-IDF vectorization

---

## 📜 License

This project is for educational purposes. Feel free to modify it for your own use.

---

## 👨‍💻 Author

- [Your Name or GitHub](https://github.com/yourusername)
