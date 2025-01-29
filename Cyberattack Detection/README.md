# Cyberattack Detection Using Machine Learning
Project Overview
The goal of this project is to develop a machine learning-based system for detecting cyberattacks by classifying network traffic data into various categories. Multiple machine learning models, including Support Vector Machine (SVM), Random Forest, Decision Tree, Naive Bayes, and XGBoost, are used to build a robust system capable of identifying anomalies and potential attacks based on packet-level features.

Key Features
Data Preprocessing:

Utilizes real-world network traffic data, cleanses it by handling missing values, and selects relevant features.
Splits the data into training and testing sets to train the models effectively.
Model Training:

Implements multiple machine learning algorithms to classify the data:
Support Vector Machine (SVM) with a linear kernel.
Random Forest Classifier to capture complex relationships.
Decision Tree Classifier for easy decision interpretation.
Naive Bayes Classifier for fast probabilistic classification.
XGBoost for high performance with large datasets.
Model Evaluation:

Evaluates model performance using accuracy, confusion matrix, and classification report.
Models are tested on a separate dataset (30% of the original data) to assess generalization.
Model Saving:

Saves trained models in joblib format for easy deployment and future use.
Objectives
Detect various types of cyberattacks (e.g., DDoS, port scanning) by analyzing network traffic data.
Compare the performance of multiple classifiers to determine the best approach for cyberattack detection.
Provide a visual representation of model performance using metrics such as accuracy and classification report.
Technologies Used
Machine Learning Libraries: Scikit-learn, XGBoost.
Data Processing: Pandas for data manipulation and preprocessing.
Visualization: Matplotlib for displaying results and confusion matrices.
Model Saving: Joblib for saving and loading models.
GUI: Tkinter for building a user interface to interact with models and results.
End Result
The project provides a user-friendly graphical interface that allows users to train and evaluate different machine learning models for cyberattack detection. The system outputs classification results along with performance summaries, including model accuracy and evaluation metrics.
