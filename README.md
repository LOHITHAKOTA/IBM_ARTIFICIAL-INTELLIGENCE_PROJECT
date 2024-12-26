# 🏨 Hotel Sentiment Analysis

This project performs **sentiment analysis** on hotel reviews using **Python**, with a focus on comparing the performance of three machine learning algorithms:  
- **Multinomial Naive Bayes**  
- **Bernoulli Naive Bayes**  
- **Logistic Regression**  

The main goal is to classify reviews as either **positive** or **negative** and evaluate the effectiveness of different algorithms.

---

## 🚀 Features

- **Predict Sentiments**: Classifies hotel reviews as positive or negative based on text content.
- **Compare Algorithms**: Evaluates the performance of Multinomial Naive Bayes, Bernoulli Naive Bayes, and Logistic Regression.
- **Scalable Design**: Easily expandable to include other algorithms or datasets.

---

## 🛠️ Project Workflow

### Step 1: Data Import
- Load the dataset containing hotel reviews.
- Explore the dataset to understand the distribution of reviews and sentiment labels.

### Step 2: Data Preprocessing
- Clean the text data:
  - Remove stopwords, special characters, and HTML tags.
  - Convert text to lowercase.
- Tokenize and lemmatize text using NLP techniques.
- Handle missing or imbalanced data if necessary.

### Step 3: Vectorization
- Convert the preprocessed text data into numerical vectors using Scikit-learn's vectorization tools:
  - **CountVectorizer**: Converts text into a matrix of word counts (Bag of Words).

### Step 4: Predictions on Reviews
- Train and test the following machine learning models:
  - **Multinomial Naive Bayes**
  - **Bernoulli Naive Bayes**
  - **Logistic Regression**
- Evaluate the models using metrics like:
  - Accuracy
  - Precision
  - Recall

---

## 📊 Technologies Used

- **Programming Language**: Python  
- **Libraries**:
  - Data Handling: Pandas, NumPy
  - NLP: NLTK
  - Vectorization & Modeling: Scikit-learn

---

## 📈 Results and Model Comparison

| Algorithm                | Accuracy | Precision | Recall | 
|--------------------------|----------|-----------|--------|
| Multinomial Naive Bayes  | 77%      | 78%       | 77%    | 
| Bernoulli Naive Bayes    | 77%      | 76%       | 78%    | 
| Logistic Regression      | 76.67%      | 80%       | 71%    | 
