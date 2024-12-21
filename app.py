import os
import streamlit as st
from PIL import Image  # For handling .webp images
import pickle

# Helper function to load images safely
def load_image(image_path, caption, use_container_width=True):
    """
    Load and display an image with error handling.

    Args:
        image_path (str): Path to the image file.
        caption (str): Caption to display below the image.
        use_container_width (bool): Whether to use the container's width.
    """
    try:
        with Image.open(image_path) as img:
            st.image(img, caption=caption, use_container_width=use_container_width)
    except FileNotFoundError:
        st.error(f"Image file not found: {image_path}")
    except Exception as e:
        st.error(f"Error loading image: {image_path}")
        st.warning(str(e))

# Streamlit app setup
st.title("💬 Twitter Gender and Sentiment Analysis")
st.subheader("Predict the gender and sentiment based on Twitter descriptions!")

# Load and display main images
st.markdown("### Explore the App")
load_image("images/main_1.webp", "Analyze gender and sentiment in one place.")
load_image("images/main.webp", "Discover insights with text analysis!")

# App description
st.write(
    """
    This app uses machine learning to analyze Twitter descriptions. It predicts the gender 
    (Male/Female) of the texter based on their input and provides sentiment analysis to 
    determine if the text conveys a positive, negative, or neutral sentiment.
    """
)

# Text input
description = st.text_area("📝 Enter Twitter Description:")

# Load model and vectorizer
model_path = os.path.join("data", "twitter_gender_model.pkl")
try:
    with open(model_path, "rb") as f:
        cv, model = pickle.load(f)
    model_loaded = True
except FileNotFoundError:
    st.error("Model file not found. Please ensure `twitter_gender_model.pkl` is in the `data/` folder.")
    model_loaded = False
except Exception as e:
    st.error(f"Error loading model: {e}")
    model_loaded = False

def preprocess_input(text, cv):
    """
    Preprocess input text for gender prediction and sentiment analysis.

    Args:
        text (str): Raw text input.
        cv (CountVectorizer): Trained vectorizer.

    Returns:
        tuple: Features for gender prediction, sentiment label.
    """
    try:
        from textblob import TextBlob  # Ensure TextBlob is imported here
        text_blob = TextBlob(text)
        sentiment_polarity = text_blob.sentiment.polarity
        sentiment_label = (
            "😊 Positive" if sentiment_polarity > 0 else "😢 Negative" if sentiment_polarity < 0 else "😐 Neutral"
        )

        processed_text = " ".join([word.lower() for word in text.split() if word.isalpha()])
        features = cv.transform([processed_text]).toarray()
        return features, sentiment_label
    except ImportError:
        logging.error("TextBlob is not installed. Please install it using `pip install textblob`.")
        raise
    except Exception as e:
        logging.error(f"Error during text preprocessing: {e}")
        return None, "Error"


def predict_gender(features):
    """
    Predict gender using the trained model.
    """
    try:
        prediction = model.predict(features)[0]
        return "👩 Female" if prediction == 1 else "👨 Male"
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return "Error"

# Analyze button
if st.button("🔍 Analyze"):
    if not model_loaded:
        st.error("Model is not loaded. Please fix the issue and try again.")
    elif not description.strip():
        st.warning("⚠️ Please enter a description.")
    else:
        features, sentiment = preprocess_input(description, cv)
        if features is not None:
            gender = predict_gender(features)
            st.markdown(f"### **Predicted Gender:** {gender}")
            st.markdown(f"### **Sentiment Analysis:** {sentiment}")
            # Load gender-specific images
            if "Female" in gender:
                load_image("images/female.webp", "👩 Female Identified!")
            elif "Male" in gender:
                load_image("images/male.webp", "👨 Male Identified!")
        else:
            st.error("⚠️ Failed to process input. Please try again.")
