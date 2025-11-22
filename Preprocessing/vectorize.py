from sklearn.feature_extraction.text import CountVectorizer
import pickle
vectorizer = CountVectorizer()
def vectorize_text(df):
    X = vectorizer.fit_transform(df['text'])

    with open('./models/vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    return X