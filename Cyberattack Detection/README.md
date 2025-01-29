# Project Title: Cyberattack Detection Using Machine Learning

Project Overview: The aim of this project is to develop a machine learning-based system for detecting cyberattacks by classifying network traffic data into various categories. It leverages multiple machine learning models, including Support Vector Machine (SVM), Random Forest, Decision Tree, Naive Bayes, and XGBoost, to build a robust system capable of identifying anomalies and potential attacks based on packet-level features.

*Key Features:*

Data Preprocessing: The project uses real-world network traffic data, cleans it by handling missing values, and selects relevant features. The data is split into training and testing sets to train the models effectively.

Model Training: Several machine learning algorithms are employed to classify the data, including:

Support Vector Machine (SVM) with a linear kernel.
Random Forest Classifier to capture complex relationships in the data.
Decision Tree Classifier for easy interpretation of decisions.
Naive Bayes Classifier for fast probabilistic classification.
XGBoost for high performance with large datasets.
Model Evaluation: The project evaluates the models' performance using accuracy, confusion matrix, and classification reports. The models are tested on a separate dataset (30% of the original data) to assess generalization.

Model Saving: After training, the models are saved in joblib format, allowing for easy deployment and future use.

Objectives:

To detect various types of cyberattacks (e.g., DDoS, port scanning, etc.) by analyzing network traffic data.
To compare the performance of multiple classifiers and determine the best approach for detecting cyberattacks.
To provide a visual representation of the model performance through metrics such as accuracy and classification report.
Technologies Used:

Machine Learning Libraries: Scikit-learn, XGBoost.
Data Processing: Pandas for data manipulation and preprocessing.
Visualization: Matplotlib for displaying results and confusion matrices.
Model Saving: Joblib for saving and loading machine learning models.
GUI: Tkinter for building a user interface where the models and results can be interacted with.
End Result:

The project provides an easy-to-use graphical interface that allows users to train different machine learning models for cyberattack detection. The system outputs classification results along with a performance summary, including the accuracy of each model.
