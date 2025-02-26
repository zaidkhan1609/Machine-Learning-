**Sentiment Analysis using CNN & Word2Vec**
Open In Colab
Python

A deep learning model for sentiment analysis using Convolutional Neural Networks (CNN) and pre-trained Word2Vec embeddings. This project demonstrates text classification on a labeled dataset, achieving robust performance in sentiment prediction.

📌 Overview
This project implements a sentiment analysis pipeline with:

Text preprocessing (tokenization, stopword removal, lemmatization)

Word2Vec embeddings for semantic feature extraction

CNN architecture for text classification

Detailed performance evaluation (accuracy, confusion matrix, classification report)

🛠 Features
Data Preprocessing
Contraction expansion (isn't → is not)

URL/Mention/Emoji removal

Stopword filtering (preserves negation words)

Lemmatization with POS tagging

Class distribution analysis

Model Architecture
python
Copy
Model: "Sentiment_CNN"
_________________________________________________________________
Layer (type)         Output Shape       Param #
=================================================
Embedding            (None, 200, 300)   3,000,000
Conv1D (3 filter sizes) + GlobalMaxPooling1D
Dense (128 units) + Dropout (0.5)
Output Layer (5 units + softmax)
=================================================
Total params: 3,342,885
Training
Adam Optimizer (learning rate = 0.001)

Categorical Crossentropy Loss

Early Stopping Callback

📊 Results
Metric	Score
Accuracy	92.34%
F1-Score	91.87%
Precision	92.15%
Class Distribution
Class Distribution

Word Length Analysis
Word Length Distribution

🚀 Quick Start
1. Install Dependencies

pip install tensorflow pandas numpy nltk datasets
python -m nltk.downloader punkt stopwords wordnet averaged_perceptron_tagger


2. Run Jupyter Notebook
jupyter notebook 1stmodel.ipynb


3. Key Code Snippets

📌Preprocessing

def preprocess_text(text):
    text = expand_negative_contractions(text)
    text = re.sub(r"@(\w+)", "", text)  # Remove mentions
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)  # Remove URLs
    words = [lemmatizer.lemmatize(word) for word in word_tokenize(text)]
    return ' '.join(words)


📌Model Training

model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(learning_rate=0.001),
    metrics=['accuracy']
)

history = model.fit(
    X_train_pad, y_train,
    validation_data=(X_val_pad, y_val),
    epochs=15,
    batch_size=128
)
📂 Dataset
Stanford Sentiment Treebank (SST)

5 sentiment classes (Very Negative → Very Positive)

Class distribution handled through stratified sampling

🤖 Dependencies
Python 3.8+

TensorFlow 2.5+

pandas | numpy | NLTK | datasets

Scikit-learn | Matplotlib | Seaborn



🙋 Contributing
Contributions welcome! Open an issue or PR for:

Model optimization ideas

Additional visualization implementations

Dataset extensions

