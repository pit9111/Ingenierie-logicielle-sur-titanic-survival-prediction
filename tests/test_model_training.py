import pytest
import pandas as pd
import os
import joblib
from sklearn.ensemble import RandomForestClassifier

# Ajouter `src/` au `sys.path` pour permettre l'import
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from model_training import load_training_data, preprocess_data, train_model, save_model

@pytest.fixture
def sample_data():
    """
    Fixture qui génère un jeu de données factice.

    Returns:
        tuple: (train_data, test_data)
    """
    train_data = pd.DataFrame({
        "Survived": [0, 1, 1, 0, 1],
        "Pclass": [3, 1, 2, 3, 1],
        "Sex": ["male", "female", "male", "male", "female"],
        "SibSp": [0, 1, 0, 2, 1],
        "Parch": [0, 0, 2, 1, 1]
    })
    test_data = pd.DataFrame({
        "PassengerId": [1, 2, 3],
        "Pclass": [3, 1, 2],
        "Sex": ["male", "female", "male"],
        "SibSp": [0, 1, 0],
        "Parch": [0, 0, 2]
    })
    return train_data, test_data

def test_preprocess_data(sample_data):
    """
    Teste si la fonction `preprocess_data` effectue correctement le prétraitement.

    Args:
        sample_data (tuple): Jeu de données simulé.
    """
    train_data, test_data = sample_data
    X, X_test, y = preprocess_data(train_data, test_data)

    assert "Sex_female" in X.columns and "Sex_male" in X.columns, "L'encodage one-hot des sexes est incorrect."
    assert len(X) == len(y), "Les dimensions des features et de la cible ne correspondent pas."

def test_train_model(sample_data):
    """
    Teste si le modèle s'entraîne correctement.

    Args:
        sample_data (tuple): Jeu de données simulé.
    """
    train_data, test_data = sample_data
    X, X_test, y = preprocess_data(train_data, test_data)

    model = train_model(X, y)

    assert isinstance(model, RandomForestClassifier), "Le modèle entraîné n'est pas un RandomForestClassifier."
    assert hasattr(model, "predict"), "Le modèle entraîné ne possède pas la méthode `predict`."

def test_save_model(sample_data, tmp_path):
    """
    Teste si le modèle est bien sauvegardé.

    Args:
        sample_data (tuple): Jeu de données simulé.
        tmp_path (Path): Répertoire temporaire pour le fichier modèle.
    """
    train_data, test_data = sample_data
    X, X_test, y = preprocess_data(train_data, test_data)

    model = train_model(X, y)

    model_file = tmp_path / "test_model.pkl"
    save_model(model, str(model_file))

    assert model_file.exists(), "Le fichier modèle n'a pas été créé."
    loaded_model = joblib.load(model_file)
    assert isinstance(loaded_model, RandomForestClassifier), "Le fichier sauvegardé ne contient pas un RandomForestClassifier."
