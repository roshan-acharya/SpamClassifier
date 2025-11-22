from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from preprocessing.load_data import load_dataset
from preprocessing.vectorize import vectorize_text
from preprocessing.clean_text import clean_text
from preprocessing.split_data import split
from preprocessing.train import train

models = [
    MultinomialNB(),
    KNeighborsClassifier(),
    GradientBoostingClassifier(),
    AdaBoostClassifier()
]

model_names = [
    'MultinomialNB',
    'KNeighborsClassifier',
    'GradientBoostingClassifier',
    'AdaBoostClassifier'
]

df=load_dataset("./Data/Raw/spam.csv")
df=clean_text(df)
X = vectorize_text(df)
y = df['label']
X_train, X_test, y_train, y_test = split(X, y)
best_model, acc = train(models, model_names, X_train, y_train)
