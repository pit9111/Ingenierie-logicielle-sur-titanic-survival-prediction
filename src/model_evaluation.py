import pandas as pd
import joblib
import os

def load_test_data(data_path: str):
    """
    Charge les données de test à partir d'un fichier CSV et applique les transformations nécessaires.

    Args:
        data_path (str): Le chemin du répertoire contenant le fichier `test.csv`.

    Returns:
        tuple: Un DataFrame contenant les données de test originales et un DataFrame transformé (`X_test`).
    """
    test_data = pd.read_csv(os.path.join(data_path, "test.csv"))
    X_test = pd.get_dummies(test_data[["Pclass", "Sex", "SibSp", "Parch"]])
    return test_data, X_test

def load_model(model_path: str):
    """
    Charge le modèle entraîné depuis un fichier.

    Args:
        model_path (str): Le chemin vers le fichier contenant le modèle entraîné (`.pkl`).

    Returns:
        object: Le modèle de machine learning chargé.
    """
    return joblib.load(model_path)

def make_predictions(model, X_test):
    """
    Génère des prédictions sur les données de test à l'aide du modèle chargé.

    Args:
        model (object): Un modèle entraîné avec une méthode `predict()`.
        X_test (DataFrame): Les caractéristiques d'entrée pour la prédiction.

    Returns:
        numpy.ndarray: Un tableau contenant les prédictions (0 ou 1).
    """
    return model.predict(X_test)

def save_predictions(test_data, predictions, output_path: str):
    """
    Sauvegarde les prédictions dans un fichier CSV.

    Args:
        test_data (DataFrame): Les données originales de test, contenant les `PassengerId`.
        predictions (numpy.ndarray): Les prédictions générées par le modèle.
        output_path (str): Le chemin où sauvegarder le fichier CSV des résultats.

    Returns:
        None
    """
    output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
    output.to_csv(output_path, index=False)
    print(f"Predictions saved in '{output_path}'")

if __name__ == "__main__":
    data_path = "../data/"
    model_path = os.path.join(data_path, "trained_model.pkl")
    output_path = os.path.join(data_path, "submission.csv")

    test_data, X_test = load_test_data(data_path)
    model = load_model(model_path)
    predictions = make_predictions(model, X_test)
    save_predictions(test_data, predictions, output_path)
    