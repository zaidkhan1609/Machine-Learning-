📰 BBC News Classification
📌 This project focuses on classifying BBC news articles into categories using machine learning techniques. The pipeline includes TF-IDF, Word2Vec, Logistic Regression, and sentiment analysis with TextBlob.

📂 Dataset
📑 The dataset consists of BBC news articles categorized into different topics.
📥 Loaded from bbc-text.csv.

⚙️ Installation
Ensure you have the required dependencies installed:

bash
Copy
Edit
pip install pandas numpy sklearn textblob gensim nltk
🔄 Workflow
✅ Data Preprocessing: Tokenization, stopword removal, TF-IDF, and Word2Vec embeddings.
✅ Model Training: Logistic Regression for classification.
✅ Evaluation: Accuracy, classification report, and sentiment analysis.

🚀 Usage
Run the Jupyter Notebook to train and evaluate the model:

python
Copy
Edit
# Load dataset
data = pd.read_csv("bbc-text.csv")

# Preprocess and train model
# (Full steps available in the notebook)
📊 Results
📌 Model performance is evaluated using accuracy and classification reports.
📌 Sentiment analysis provides insights into article tones.

🔮 Future Improvements
✨ Implement deep learning models like LSTMs or BERT.
✨ Explore additional feature engineering techniques.
