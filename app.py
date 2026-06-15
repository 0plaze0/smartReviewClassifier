import streamlit as st
import pickle

# Page Configuration
st.set_page_config(page_title="Smart Review Classifier")

st.title("Smart Review Classifier")
st.write("An end-to-end Machine Learning pipeline analyzing e-commerce product sentiment in real-time.")
st.markdown("---")

# Load the saved model artifacts safely
@st.cache_resource
def load_pipeline():
    with open('vectorizer.pkl', 'rb') as f:
        vec = pickle.load(f)
    with open('model.pkl', 'rb') as f:
        mod = pickle.load(f)
    return vec, mod

try:
    vectorizer, model = load_pipeline()
    
    # User Input Interface
    user_review = st.text_area(
        "Paste an Amazon Product Review below:", 
        placeholder="Type your product feedback here..."
    )

    if st.button("Analyze Sentiment", type="primary"):
        if user_review.strip() == "":
            st.warning("Please enter some text to analyze.")
        else:
            # Transform text and predict
            vec_input = vectorizer.transform([user_review])
            prediction = model.predict(vec_input)[0]
            probabilities = model.predict_proba(vec_input)[0]
            
            # Display results
            st.markdown("### Prediction Result")
            if prediction == 1:
                st.success(f"**Positive Sentiment** (Confidence: {probabilities[1]:.2%})")
            else:
                st.error(f"**Negative Sentiment** (Confidence: {probabilities[0]:.2%})")
                
except FileNotFoundError:
    st.error("Model artifacts not found! Please run `python train.py` first to generate the model files.")