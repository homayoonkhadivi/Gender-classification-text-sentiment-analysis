import os
import pandas as pd
import re
import logging
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import nltk
import pickle
import chardet

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Ensure NLTK resources are available
try:
    nltk.download('punkt')
    nltk.download('wordnet',)
    nltk.download('stopwords')
    nltk.download('punkt_tab')
    logging.info("NLTK resources downloaded successfully.")
except Exception as e:
    logging.error(f"Failed to download NLTK resources: {e}")
    raise

class TwitterGenderClassifier:
    """
    A class for gender classification based on Twitter user descriptions.
    Includes preprocessing, feature extraction, model training, and evaluation.
    """
    def __init__(self, max_features=5000):
        logging.info("Initializing the TwitterGenderClassifier.")
        self.cv = CountVectorizer(max_features=max_features, stop_words='english')
        self.model = RandomForestClassifier()

    def preprocess_text(self, text):
        """
        Clean and preprocess text by removing non-alphabetic characters, converting to lowercase,
        tokenizing, and lemmatizing.

        Args:
            text (str): Raw text input.

        Returns:
            str: Processed text.
        """
        try:
            text = re.sub("[^a-zA-Z]", " ", text)
            text = text.lower()
            words = word_tokenize(text)
            lemma = WordNetLemmatizer()
            words = [lemma.lemmatize(word) for word in words if word not in stopwords.words('english')]
            return " ".join(words)
        except Exception as e:
            logging.error(f"Error during text preprocessing: {e}")
            return ""

    def preprocess_data(self, df):
        """
        Apply text preprocessing to the 'description' column of a DataFrame.

        Args:
            df (DataFrame): DataFrame containing the data.

        Returns:
            DataFrame: Processed DataFrame.
        """
        logging.info("Preprocessing data...")
        df['description'] = df['description'].apply(self.preprocess_text)
        return df

    def feature_extraction(self, texts):
        """
        Extract features from text using CountVectorizer.

        Args:
            texts (list): List of preprocessed text.

        Returns:
            ndarray: Feature matrix.
        """
        logging.info("Extracting features using CountVectorizer.")
        return self.cv.fit_transform(texts).toarray()

    def train(self, x_train, y_train):
        """
        Train the RandomForestClassifier model.

        Args:
            x_train (ndarray): Training feature matrix.
            y_train (ndarray): Training target vector.
        """
        logging.info("Training the RandomForestClassifier model.")
        self.model.fit(x_train, y_train)

    def evaluate(self, x_test, y_test):
        """
        Evaluate the trained model on test data.

        Args:
            x_test (ndarray): Test feature matrix.
            y_test (ndarray): Test target vector.

        Returns:
            float: Accuracy score.
        """
        logging.info("Evaluating the model.")
        try:
            y_pred = self.model.predict(x_test)
            accuracy = 100.0 * accuracy_score(y_test, y_pred)
            logging.info(f"Model accuracy: {accuracy:.2f}%")
            return accuracy
        except Exception as e:
            logging.error(f"Error during model evaluation: {e}")
            return 0.0

    def save_model(self, filepath):
        """
        Save the trained model and vectorizer to a file.

        Args:
            filepath (str): Path to save the model.
        """
        logging.info("Saving the model to disk.")
        try:
            with open(filepath, 'wb') as f:
                pickle.dump((self.cv, self.model), f)
        except Exception as e:
            logging.error(f"Error saving the model: {e}")

# Main workflow
if __name__ == "__main__":
    try:
        logging.info("Loading data from 'data' folder...")
        csv_path = r"D:\perosnal_project\Gender_classification_text_sentiment_analysis\data\gender-classifier-DFE-791531.csv"


        # Detect file encoding
        with open(csv_path, "rb") as f:
            result = chardet.detect(f.read())
            encoding = result['encoding']
            logging.info(f"Detected file encoding: {encoding}")

        # Load data with detected encoding
        data = pd.read_csv(csv_path, encoding=encoding)
        logging.info(f"Loaded data with shape: {data.shape}")

        # Preprocess the dataset
        data = pd.concat([data.gender, data.description], axis=1).dropna()
        data.gender = [1 if gender == "female" else 0 for gender in data.gender]

        classifier = TwitterGenderClassifier()
        data = classifier.preprocess_data(data)

        description_list = data['description']
        if description_list.isnull().all():
            raise ValueError("No valid descriptions for training.")

        x = classifier.feature_extraction(description_list)
        y = data['gender']
        print(y)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)

        classifier.train(x_train, y_train)
        accuracy = classifier.evaluate(x_test, y_test)
        logging.info(f"Final Model Accuracy: {accuracy:.2f}%")

        # Save model
        model_path = os.path.join("data", "twitter_gender_model.pkl")
        classifier.save_model(model_path)
        logging.info("Model training and saving completed successfully.")
    except Exception as e:
        logging.error(f"Error in main workflow: {e}")
