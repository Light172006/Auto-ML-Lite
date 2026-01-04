# 🚀 AutoML Lite – Streamlined Machine Learning with Streamlit

AutoML Lite is a lightweight automated machine learning web app built using Streamlit.
It allows users to train high-quality ML models without writing any code, while still following proper ML workflows like data cleaning, feature engineering, encoding, and hyperparameter tuning.

This project is designed to bridge the gap between ease of use and real ML best practices.

## ✨ Key Features

📂 Upload any tabular dataset (CSV)

🎯 Select target feature dynamically

🔁 Choose model type:

        Regression
        Classification

🧹 Automatic data cleaning

🛠 Feature engineering pipeline

🔢 One-Hot Encoding (applied only when required)

⚙️ Hyperparameter tuning on multiple models

🏆 Best model selection based on performance

📊 Displays final evaluation score

💾 Download trained model as a .pkl file

## 🧠 How It Works

User uploads dataset

Selects target column

Chooses model type (Regression / Classification)

Clicks the “Train Model” button

The app then automatically:

Cleans missing and inconsistent data

Performs feature engineering

Applies One-Hot Encoding for categorical variables

Trains multiple ML models

Performs hyperparameter tuning using predefined custom functions

Evaluates all models

Best performing model is selected

Model score is displayed

Trained model is available for download

## 📦 Tech Stack

Python

Streamlit

Pandas

NumPy

Scikit-learn

Joblib / Pickle

## 🧪 Supported Models
Regression

    Linear Regression

    Ridge / Lasso

    Random Forest Regressor

    KNeighborsRegressor

Classification

    Logistic Regression

    Random Forest Classifier

    BernoulliNB

    GaussianNB

    KNeighborsClassifier

Hyperparameters are tuned using custom predefined functions written from scratch.

## 🎯 Why AutoML Lite?

Built for learning + real-world usability

No black-box AutoML — everything is transparent

Easy to extend with new models

Ideal for:

Students

ML beginners

Rapid prototyping

Hackathons

## 🚧 Limitations (Honest Note)

Works best with clean tabular data

Not intended for deep learning or large-scale datasets

Feature engineering is rule-based (not automated feature discovery)

## 🔮 Future Improvements

Add cross-validation visualization

Support for pipelines export

Model explainability (SHAP)

Train/Test split customization

Cloud deployment

## 🙌 Author

Built with curiosity and consistency by Light

B.Tech CSE (AI & ML) | Self-learning ML Practitioner
