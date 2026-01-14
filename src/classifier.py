from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np

def train_model(X, y, algorithm='KNN', k=3):
    """
    Train a classifier model.
    """
    if algorithm == 'KNN':
        model = KNeighborsClassifier(n_neighbors=k)
    elif algorithm == 'SVM':
        model = SVC(probability=True, kernel='linear') # Linear kernel usually good for high dim
    else:
        raise ValueError("Unknown algorithm")
    
    # Split data for internal validation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model.fit(X_train, y_train)
    
    # Evaluate on the test split
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    return model, acc, cm, (X_test, y_test)

def predict_image(model, feature_vector):
    """
    Predict class for a single feature vector.
    """
    # Reshape for single sample
    feature_vector = feature_vector.reshape(1, -1)
    prediction = model.predict(feature_vector)
    probabilities = model.predict_proba(feature_vector) if hasattr(model, "predict_proba") else None
    
    return prediction[0], probabilities
