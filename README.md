# SpamClassifier
This project uses Logistic Regression to classify messages as either "spam" or "ham" (non-spam). A Streamlit user interface (UI) allows users to input a message and get a real-time prediction of whether it's spam or ham.

# Technologies Used

Python 
Streamlit - for building the web UI.

Scikit-learn - for machine learning algorithms and data preprocessing.

Pandas - for data manipulation and analysis.

NumPy - for handling numerical operations.

Pickle - for saving and loading the trained model.

# How It Works

Preprocessing: The raw text messages are preprocessed by:
               Lowercasing the text
               Removing stopwords, punctuation, and special characters
               Converting the text into numeric features using TF-IDF (Term Frequency-Inverse Document Frequency) vectorization.

Model Training: The Logistic Regression model is trained on a labeled dataset of spam and ham messages. The model learns to differentiate between spam and ham messages based                  on the features extracted from the text.

Prediction: The trained model is used to predict whether an input message is spam or ham. The user inputs a message into the Streamlit UI, and the model returns the                       classification.

