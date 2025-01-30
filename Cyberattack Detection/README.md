# 🔥 Cyberattack Detection Using Machine Learning 🔥
🚀 Project Overview
This project aims to build a machine learning-based system for detecting cyberattacks by classifying network traffic data. It employs multiple machine learning models to identify anomalies and potential threats based on packet-level features.

📌 Features
✔️ Data Preprocessing 🛠️

Cleans network traffic data by handling missing values
Selects relevant features for better model performance
Splits data into training (70%) and testing (30%) sets
✔️ Model Training 🤖
The project trains multiple machine learning models:

Support Vector Machine (SVM) 📈 – Linear kernel for efficient classification
Random Forest Classifier 🌳 – Captures complex relationships in the data
Decision Tree Classifier 🌲 – Simple and interpretable decision-making
Naive Bayes Classifier 🎲 – Fast probabilistic classification
XGBoost ⚡ – Optimized for large datasets
✔️ Model Evaluation 📊

Assesses models using accuracy, confusion matrix, and classification report
Evaluates performance on an independent test dataset (30%)
✔️ Model Saving & Deployment 💾

Saves trained models in Joblib format for easy deployment
Allows reusability of trained models without retraining
✔️ User Interface (GUI) 🖥️

Built using Tkinter for easy interaction
Enables users to select a model and test its performance
🎯 Project Objectives
🎯 Detect different types of cyberattacks (DDoS, port scanning, etc.) using network traffic data.
🎯 Compare the performance of different machine learning classifiers.
🎯 Provide a visual performance summary using accuracy and classification reports.

🔧 Technologies Used
🛠️ Machine Learning Libraries: Scikit-learn, XGBoost
🛠️ Data Processing: Pandas, NumPy
🛠️ Visualization: Matplotlib (for displaying results & confusion matrices)
🛠️ Model Persistence: Joblib (for saving & loading trained models)
🛠️ GUI Development: Tkinter

