import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string


nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('punkt')

def Remove_unnecessary_column(df):
    df=df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'])
    df = df.rename(columns={'v1':'label', 'v2':'text'})
    return df

def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))

def to_lowercase(text):
    return text.lower()

#chatword treatment
chat_words = {
    "u": "you",
    "ur": "your",
    "4": "for",
    "2": "to",
    "btw": "by the way",
    "idk": "I don't know",
    "imo": "in my opinion",
    "omg": "oh my god",
    "ttyl": "talk to you later",
    "brb": "be right back",
    "lmk": "let me know",
    "smh": "shaking my head",
    "tbh": "to be honest",
    "fyi": "for your information",
    "np": "no problem",
    "thx": "thanks",
    "pls": "please"
}

def replace_chat_words(text):
    words = text.split()
    new_words = [chat_words.get(word, word) for word in words]
    return ' '.join(new_words)

stop_words = set(stopwords.words('english'))

def remove_stop_words(text):
    words = word_tokenize(text)
    filtered_words = []
    for word in words:
        if word not in stop_words:
            filtered_words.append(word)
    return ' '.join(filtered_words)

def clean_text(df):
    df = Remove_unnecessary_column(df)
    df['text'] = df['text'].apply(to_lowercase)
    df['text'] = df['text'].apply(remove_punctuation)
    df['text'] = df['text'].apply(replace_chat_words)
    df['text'] = df['text'].apply(remove_stop_words)
    df.to_csv('../Data/Processed/cleaned_spam.csv', index=False)
    return df

if __name__ == "__main__":
    df = pd.read_csv('../Data/Raw/spam.csv', encoding='latin-1')
    cleaned_df = clean_text(df)
    print(cleaned_df.head())







