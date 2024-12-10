import os
import json
import datetime
import csv
import nltk
import ssl
import streamlit as st
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ssl._create_default_https_context = ssl._create_unverified_context
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download('punkt')

# Set the page title and other configurations here
st.set_page_config(page_title="L.U.N.A.", page_icon=":moon:", layout="wide")

# Load intents from the JSON file
file_path = os.path.abspath("./intents.json")
with open(file_path, "r") as file:
    intents = json.load(file)



# Create the vectorizer and classifier
vectorizer = TfidfVectorizer(ngram_range=(1, 4))
clf = LogisticRegression(random_state=0, max_iter=10000)



# Preprocess the data
tags = []
patterns = []
for intent in intents:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)



# training the model
x = vectorizer.fit_transform(patterns)
y = tags
clf.fit(x, y)

def chatbot(input_text):
    input_text = vectorizer.transform([input_text])
    tag = clf.predict(input_text)[0]
    for intent in intents:
        if intent['tag'] == tag:
            response = random.choice(intent['responses'])
            return response
        
counter = 0



def main():

    st.markdown("""
    <style>
    /* Main background and text color */
    .main {
        background-color: white !important;
        color: black !important;
    }

    /* Sidebar background color */
    .sidebar .sidebar-content {
        background-color: #f0f0f0 !important;  /* Light gray */
    }

    /* Sidebar text color */
    .sidebar .sidebar-content * {
        color: black !important;
    }

    /* Textbox background and text color */
    .stTextInput input, .stTextArea textarea {
        background-color: #f9f9f9 !important;  /* Light gray */
        color: black !important;
    }
                
    /* Styling for the placeholder text */
    .stTextInput input::placeholder, .stTextArea textarea::placeholder {
        color: gray !important;  /* Adjust this color as needed */
    }

    /* Remove Streamlit footer */
    footer {
        visibility: hidden;
    }
    </style>
""", unsafe_allow_html=True)
    

    

    global counter
    st.title("L.U.N.A.")

    # Create a sidebar menu with options
    menu = ["HOME", "CONVERSATION HISTORY", "ABOUT"]
    choice = st.sidebar.selectbox("MENU", menu)


    # Home Menu
    if choice == "HOME":
        st.write("Welcome..")
        st.write("What can I help with?")

        # Check if the chat_log.csv file exists, and if not, create it with column names
        if not os.path.exists('chat_log.csv'):
            with open('chat_log.csv', 'w', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(['User Input', 'Chatbot Response', 'Timestamp'])


        counter += 1
        
        # Add a placeholder text in the input box
        user_input = st.text_input("You:", key=f"user_input_{counter}", placeholder="Please type a message and press Enter to start the conversation")

        
        if user_input:

            # Convert the user input to a string
            user_input_str = str(user_input)

            #get the chatbot response
            response = chatbot(user_input)
            st.text_area("Reply:", value=response, height=120, max_chars=None, key=f"chatbot_response_{counter}")

            # Get the current timestamp
            timestamp = datetime.datetime.now().strftime(f"%Y-%m-%d %H:%M:%S")

            # Save the user input and chatbot response to the chat_log.csv file
            with open('chat_log.csv', 'a', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([user_input_str, response, timestamp])

            if response.lower() in ['goodbye', 'bye']:
                st.write("Thank you for chatting with me. Have a great day!")
                st.stop()



    if choice == "CONVERSATION HISTORY":
         st.header("CONVERSATION HISTORY")

    # Use an expander to display conversation history
         with st.expander("Click to see Conversation History"):
             try:
                 with open('chat_log.csv', 'r', encoding='utf-8') as csvfile:
                     csv_reader = csv.reader(csvfile)
                     next(csv_reader)  # Skip the header row

                # Iterate through rows
                     for row in csv_reader:
                         if len(row) < 3:  # Check if the row has fewer than 3 columns
                        # Log the issue to console and skip the row
                             print(f"Skipping incomplete or malformed row: {row}")
                             continue

                    # Display the conversation details
                         st.text(f"User: {row[0]}")
                         st.text(f"Chatbot: {row[1]}")
                         st.text(f"Timestamp: {row[2]}")
                         st.markdown("---")
             except FileNotFoundError:
                 st.error("The file 'chat_log.csv' was not found.")
             except Exception as e:
                 st.error(f"An unexpected error occurred: {e}")



    elif choice == "ABOUT":
        st.write("The goal of this project is to create a chatbot that can understand and respond to user input based on intents. The chatbot is built using Natural Language Processing (NLP) library and Logistic Regression, to extract the intents and entities from user input. The chatbot is built using Streamlit, a Python library for building interactive web applications.")

        st.subheader("PROJECT OVERVIEW:")

        st.write("""
        The project is divided into two parts:
        1. NLP methods and the Logistic Regression algorithm are utilized to train the chatbot on labeled intents and entities.
        2. The chatbot interface is developed using the Streamlit web framework, which creates a web-based platform where users can input text and receive responses from the chatbot.
        """)

        st.subheader("DATASET:")

        st.write("""
        The dataset used in this project is a collection of labelled intents and entities. The data is stored in a list.
        - Intents: The intent of the user input (e.g. "greeting", "budget", "about")
        - Entities: The entities extracted from user input (e.g. "Hi", "How do I create a budget?", "What is your purpose?")
        - Text: The user input text.
        """)

        st.subheader("STREAMLIT CHATBOT INTERFACE:")

        st.write("The chatbot interface is built using Streamlit. The interface includes a text input box for users to input their text and a chat window to display the chatbot's responses. The interface uses the trained model to generate responses to user input.")

        st.subheader("CONCLUSION:")

        st.write("In this project, a chatbot is built that can understand and respond to user input based on intents. The chatbot was trained using NLP and Logistic Regression, and the interface was built using Streamlit. This project can be extended by adding more data, using more sophisticated NLP techniques, deep learning algorithms.")

if __name__ == '__main__':
    main()
