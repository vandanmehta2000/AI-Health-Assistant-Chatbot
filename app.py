import streamlit as st
from transformers import pipeline
import nltk
import torch
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


nltk.download('stopwords')
nltk.download('punkt_tab')

model_name_or_path = "m42-health/Llama3-Med42-8B"
chatbot = pipeline("text-generation", model=model_name_or_path, torch_dtype = torch.bfloat16, device_map = "auto")

# pre-process user input
def preprocess_input(user_input):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(user_input)
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)

# Define healthcare specific response logic
def healthcare_chatbot(user_input):
    user_input = preprocess_input(user_input).lower()
    if "symptom" in user_input:
        return "It seems like you are experiencing symptoms. Consult a doctor if symptoms persis."
    elif "appointment" in user_input:
        return "Would you like to schedule an appointment with a doctor?"
    elif "medication" in user_input:
        return "It's important to take your prescribed medications regularly. If you have concerns, consult a doctor."
    else:
        response = chatbot(user_input, max_length = 512, num_return_sequences = 1, temperature = 0.4, top_k = 150, top_p = 0.75)
        return response[0]['generated_text'][len(user_input) :]


# Streamlit Web App Interface
def main():
    st.title("Healthcare Assistant Chatbot")
    user_input = st.text_input("How can I assist you today?", "")
    if st.button("Submit"):
        if user_input:
            st.write("User: ", user_input)
            with st.spinner("Processing your query ... Please wait"):
                response = healthcare_chatbot(user_input)
            st.write("Healthcare Assistant: ", response)
            print(response)
        else:
            st.write("Please enter a query.")

if __name__ == "__main__":
    main()
