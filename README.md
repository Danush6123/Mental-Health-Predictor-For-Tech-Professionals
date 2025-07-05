# 🧠 Mental Health Monitor for Tech Employees

This project is a Mental Health Monitoring Web App specifically designed for tech employees. It leverages machine learning to predict mental health risk based on user responses to a survey and provides a user-friendly interface for interaction.

## 💡 Features

- Collects mental health-related survey data
- Predicts if a user is at risk using a trained ML model
- Frontend built with HTML/CSS
- Backend powered by Flask
- Clean and responsive UI for easy interaction

## 📁 Project Structure

```

mental\_health/
├── app.py                       # Flask app entry point
├── model\_creation.py           # ML model training script
├── mental\_health\_model.joblib  # Trained ML model
├── target\_encoder.joblib       # Encoded label transformer
├── survey.csv                  # Dataset used for model training
├── static/
│   └── style.css               # CSS styling
│   └── images/
│       └── background.jpg      # Background image
└── templates/
└── index.html              # Main HTML page

````

## 🚀 Getting Started

### 🔧 Prerequisites

Ensure you have the following installed:

- Python 3.7+
- pip

### 📦 Installation

```bash
git clone https://github.com/yourusername/mental_health_monitor.git
cd mental_health_monitor/mental_health
pip install -r requirements.txt
````

> *Note: You may have to create `requirements.txt` with necessary packages like `Flask`, `pandas`, `scikit-learn`, `joblib`, etc.*

### ▶️ Run the App

```bash
python app.py
```

Navigate to `http://127.0.0.1:5000/` in your browser.

## 🧠 Model

* Algorithm: Logistic Regression (or similar)
* Input: Survey responses (features like work environment, mental health history, etc.)
* Output: Binary prediction - At Risk / Not at Risk

## 📊 Dataset

* Source: `survey.csv`
* Preprocessed and encoded with `target_encoder.joblib`

## 🙌 Acknowledgments

* [Kaggle](https://www.kaggle.com/) for dataset inspiration
* [Flask](https://flask.palletsprojects.com/) for the backend framework

## 💻 Developer

**Danush G** -
https://github.com/Danush6123


