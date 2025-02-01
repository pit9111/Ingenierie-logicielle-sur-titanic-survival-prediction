import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# Définir le chemin des données
data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))

def load_training_data():
    """
    Charge les fichiers CSV des données d'entraînement et de test.

    Returns:
        tuple: (train_data, test_data) - DataFrames des jeux de données.
    """
    train_data = pd.read_csv(os.path.join(data_path, "train.csv"))
    test_data = pd.read_csv(os.path.join(data_path, "test.csv"))
    return train_data, test_data

def preprocess_data(train_data, test_data):
    """
    Prépare les données en appliquant le one-hot encoding aux variables catégorielles.

    Args:
        train_data (DataFrame): Jeu de données d'entraînement.
        test_data (DataFrame): Jeu de données de test.

    Returns:
        tuple: (X, X_test, y) - Features encodées et cible.
    """
    y = train_data["Survived"]
    features = ["Pclass", "Sex", "SibSp", "Parch"]
    X = pd.get_dummies(train_data[features])
    X_test = pd.get_dummies(test_data[features])
    return X, X_test, y

def train_model(X, y):
    """
    Entraîne un modèle RandomForestClassifier.

    Args:
        X (DataFrame): Features d'entraînement.
        y (Series): Variable cible.

    Returns:
        RandomForestClassifier: Modèle entraîné.
    """
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
    model.fit(X, y)
    return model

def save_model(model, model_filename="trained_model.pkl"):
    """
    Sauvegarde le modèle entraîné sous forme d'un fichier pickle.

    Args:
        model (RandomForestClassifier): Modèle entraîné.
        model_filename (str, optional): Nom du fichier à sauvegarder. Defaults to "trained_model.pkl".
    """
    model_path = os.path.join(data_path, model_filename)
    joblib.dump(model, model_path)
    print(f"Model training completed and saved in '{model_path}'")

if __name__ == "__main__":
    # Charger les données
    train_data, test_data = load_training_data()

    # Prétraitement des données
    X, X_test, y = preprocess_data(train_data, test_data)

    # Entraîner le modèle
    model = train_model(X, y)

    # Sauvegarder le modèle
    save_model(model)
