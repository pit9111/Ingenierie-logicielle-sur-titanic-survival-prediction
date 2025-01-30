import pandas as pd
import joblib
import os

def load_test_data(data_path: str):
    """Charge les données de test et applique les transformations nécessaires."""
    test_data = pd.read_csv(os.path.join(data_path, "test.csv"))
    X_test = pd.get_dummies(test_data[["Pclass", "Sex", "SibSp", "Parch"]])
    return test_data, X_test

def load_model(model_path: str):
    """Charge le modèle entraîné."""
    return joblib.load(model_path)

def make_predictions(model, X_test):
    """Effectue des prédictions à partir du modèle chargé."""
    return model.predict(X_test)

def save_predictions(test_data, predictions, output_path: str):
    """Sauvegarde les prédictions dans un fichier CSV."""
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