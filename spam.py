import streamlit as st
import pickle

# 1. Model āṇi Vectorizer load karā
model = pickle.load(open('spam_model.pkl', 'rb'))
cv = pickle.load(open('vectorizer.pkl' , 'rb'))

# Page configuration
st.set_page_config(page_title="SMS Spam Detector", page_icon="✉️")

# UI Design
st.title("✉️ SMS Spam Detection System")
st.write("Type your message below and check whether it is Spam or Ham.")

# User Input
input_sms = st.text_area("Enter the message", height=150)

if st.button('Predict'):

    if input_sms.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        # 1. Preprocess (Vectorize)
        data = cv.transform([input_sms])

        # 2. Predict
        prediction = model.predict(data)[0]

        # 3. Result Display
        if prediction == True:
            st.error("🚨 This is a SPAM message!")
        else:
            st.success("✅ This is a HAM (Normal) message.")

# Footer
st.markdown("---")
st.caption("Created by Atharv Taral")
st.caption("Powered by Multinomial Naive Bayes Model")

