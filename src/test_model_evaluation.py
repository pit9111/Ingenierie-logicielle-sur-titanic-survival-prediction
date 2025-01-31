import pytest
import pandas as pd
import numpy as np
from model_evaluation import load_test_data, make_predictions, save_predictions

class FakeModel:
    """
    Une classe simulant un modèle entraîné avec une méthode `predict()`.
    
    Méthodes:
        predict(X): Retourne un tableau de zéros ayant la même longueur que `X`.
    """
    def predict(self, X):
        return np.zeros(len(X))  # Retourne uniquement des zéros pour tester la fonction

@pytest.fixture
def sample_test_data():
    """
    Fixture qui génère un jeu de données de test factice.

    Returns:
        tuple: Un DataFrame avec des données simulées et un DataFrame transformé (`X_test`).
    """
    test_data = pd.DataFrame({
        "PassengerId": [1, 2, 3],
        "Pclass": [3, 1, 2],
        "Sex": ["male", "female", "male"],
        "SibSp": [0, 1, 0],
        "Parch": [0, 0, 2]
    })
    
    X_test = pd.get_dummies(test_data[["Pclass", "Sex", "SibSp", "Parch"]])
    return test_data, X_test

def test_load_test_data(tmp_path):
    """
    Teste que la fonction `load_test_data` charge correctement les données et applique les transformations nécessaires.

    Vérifications :
    - Le fichier CSV est bien chargé.
    - Les colonnes encodées ("Sex_female" et "Sex_male") existent.
    - `X_test` a le même nombre de lignes que `test_data`.

    Args:
        tmp_path (Path): Un répertoire temporaire fourni par pytest.

    Returns:
        None
    """
    file_path = tmp_path / "test.csv"
    sample_data = """PassengerId,Pclass,Sex,SibSp,Parch
1,3,male,0,0
2,1,female,1,0
3,2,male,0,2
"""
    file_path.write_text(sample_data)
    
    test_data, X_test = load_test_data(str(tmp_path))
    
    assert not test_data.empty, "Les données ne doivent pas être vides."
    assert "Sex_female" in X_test.columns and "Sex_male" in X_test.columns, "Les colonnes encodées doivent être présentes."
    assert len(test_data) == len(X_test), "X_test doit avoir le même nombre de lignes que test_data."

def test_make_predictions(sample_test_data):
    """
    Teste que la fonction `make_predictions` retourne bien un tableau de la bonne taille.

    Vérifications :
    - Le nombre de prédictions doit être égal au nombre d'entrées dans `X_test`.
    - Les prédictions ne doivent contenir que des valeurs 0 ou 1.

    Args:
        sample_test_data (tuple): Une fixture contenant des données de test simulées.

    Returns:
        None
    """
    _, X_test = sample_test_data
    fake_model = FakeModel()  # Utilisation de la classe FakeModel

    predictions = make_predictions(fake_model, X_test)

    assert len(predictions) == len(X_test), "Le nombre de prédictions ne correspond pas à la taille de X_test."
    assert np.all((predictions == 0) | (predictions == 1)), "Les prédictions doivent être 0 ou 1."

def test_save_predictions(tmp_path, sample_test_data):
    """
    Teste que la fonction `save_predictions` génère bien un fichier CSV valide.

    Vérifications :
    - Le fichier généré ne doit pas être vide.
    - Il doit contenir les colonnes "PassengerId" et "Survived".
    - Le nombre de lignes doit être égal au nombre de prédictions.

    Args:
        tmp_path (Path): Un répertoire temporaire fourni par pytest.
        sample_test_data (tuple): Une fixture contenant des données de test simulées.

    Returns:
        None
    """
    test_data, _ = sample_test_data
    predictions = np.array([1, 0, 1])
    
    output_file = tmp_path / "submission.csv"
    save_predictions(test_data, predictions, str(output_file))
    
    saved_df = pd.read_csv(output_file)
    
    assert not saved_df.empty, "Le fichier de sortie ne doit pas être vide."
    assert list(saved_df.columns) == ["PassengerId", "Survived"], "Les colonnes du fichier doivent être 'PassengerId' et 'Survived'."
    assert len(saved_df) == len(predictions), "Le fichier de sortie doit contenir autant de lignes que de prédictions."