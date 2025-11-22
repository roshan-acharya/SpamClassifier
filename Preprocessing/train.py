import numpy as np
import pickle
from sklearn.model_selection import cross_val_score, StratifiedKFold

def train(models, model_names, X_train, y_train, save_path="./models/best_spam_model.pkl"):
    model_accuracy = {}

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for i, model in enumerate(models):
        model.fit(X_train, y_train)
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
        accuracy = np.mean(scores)
        model_accuracy[model_names[i]] = accuracy
        print(f"{model_names[i]} Accuracy: {accuracy:.4f}")

    best_model_name = max(model_accuracy, key=model_accuracy.get)
    best_model_index = model_names.index(best_model_name)
    best_model = models[best_model_index]
    print(f"Best Model: {best_model_name} with Accuracy: {model_accuracy[best_model_name]:.4f}")

    with open(save_path, 'wb') as f:
        pickle.dump(best_model, f)

    return best_model, model_accuracy
