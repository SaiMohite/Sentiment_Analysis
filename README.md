# Sentiment_Analysis
# COMPANY : CODTECH IT SOLUTIONS
# NAME : MOHITE SAI DATTATRAY
# INTERN ID : CT6WPYG
# DOMAIN : MACHINE LEARNING
# DURATION : 6 WEEKS
# MENTOR : NEELA SANTHOSH KUMAR
# This project aims to build a sentiment analysis model that classifies restaurant reviews into positive and negative sentiments. It utilizes Natural Language Processing (NLP) techniques to preprocess text data and a machine learning model to predict the sentiment based on the processed data. The main goal is to achieve high accuracy in classifying customer reviews, enabling insights into customer satisfaction and areas for improvement.
# Dataset :
The dataset used for this project is a CSV file titled "European Restaurant Reviews", containing: Review: Textual data representing customer feedback on restaurants. Sentiment: The sentiment label (Positive/Negative) corresponding to the customer review.
# Key Steps :
1.Data Exploration: Displayed a preview of the dataset, focusing on the Review and Sentiment columns. Counted the number of positive and negative reviews to understand the class distribution.

2.Data Preprocessing:
  i.Cleaning Text: Removed special characters, numbers, and extra spaces. Converted text to lowercase for uniformity.
  ii.Tokenization and Stemming: Tokenized words using nltk's word_tokenize. Applied stemming using the Porter Stemmer to reduce words to their root form.
  
3.Label Encoding: Encoded the sentiment labels into numeric values (e.g., Positive = 1, Negative = 0) using LabelEncoder to ensure compatibility with machine learning algorithms.

4.TF-IDF Vectorization: Transformed textual data into numerical features using TF-IDF (Term Frequency-Inverse Document Frequency). Configured the vectorizer to consider unigrams and bigrams, with a maximum feature size of 8000.

5.Handling Class Imbalance: Computed class weights dynamically using sklearn's compute_class_weight function to address any imbalance in the dataset.

6.Model Training: Trained a Logistic Regression model with class weights to classify the reviews. Configured the model with a high number of iterations (max_iter=1000) to ensure convergence.

7.Model Evaluation: Evaluated the model using the test set and reported: Accuracy: The overall percentage of correctly classified reviews. Classification Report: Precision, recall, and F1-score for each sentiment class. Confusion Matrix: A matrix showing correct and incorrect classifications.

8.Saving the Model and Vectorizer: Saved the trained model and TF-IDF vectorizer as pickle files (logistic_regression_model.pkl and tfidf_vectorizer.pkl) for future use.

9.Prediction Function: Created a function to predict the sentiment of a single review: Cleans and preprocesses the input review. Transforms the review using the saved TF-IDF vectorizer. Predicts the sentiment using the trained model. Converts the numeric prediction back to the original sentiment label.

# Tools and Technologies
1.Python Libraries: pandas and numpy: For data manipulation and analysis. scikit-learn: For model training, evaluation, and vectorization. nltk: For NLP tasks such as tokenization and stemming. pickle: For saving and loading the model and vectorizer.

2.Dataset Preprocessing: Regular expressions for text cleaning. TF-IDF for feature extraction.

3.Machine Learning Model: Logistic Regression with class weights to handle imbalanced data.
