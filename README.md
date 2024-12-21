# Twitter Gender Classification and Sentiment Analysis

This project uses Natural Language Processing (NLP) and machine learning to classify the gender of Twitter users based on their descriptions. Additionally, it includes sentiment analysis for further insights into the text.

## **Features**
- Preprocessing of textual data (tokenization, lemmatization, and stop-word removal).
- Implementation of machine learning models:
  - Random Forest
  - Logistic Regression
  - XGBoost
- Accuracy evaluation for gender classification.
- Sentiment analysis using `TextBlob`.

## **Technologies Used**
- Python
- NLTK
- Scikit-learn
- XGBoost
- Streamlit

## **Steps**
1. Preprocess Twitter user descriptions.
2. Extract features using the Bag of Words (BoW) method.
3. Train and evaluate machine learning models for gender classification.
4. Save the trained model.
5. Deploy the model in a **Streamlit** app to predict gender and perform sentiment analysis.

## **How to Use**
1. Clone this repository.
2. Install the required libraries:
   ```
   pip install -r requirements.txt
   ```
3. Run the train_model.py script to train and save the model.
4. Launch the Streamlit app:
   ```
   streamlit run app.py
   ```

### Sentiment Analysis

**Sentiments are calculated for user descriptions as positive, neutral, or negative using the TextBlob library.**
