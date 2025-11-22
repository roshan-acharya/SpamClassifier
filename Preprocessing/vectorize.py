from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
def vectorize_text(df):
    X = vectorizer.fit_transform(df['text'])
    return X