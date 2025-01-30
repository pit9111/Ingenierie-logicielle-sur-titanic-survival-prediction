import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from joblib import dump

def train_and_save_model(train_data_path: str, test_data_path: str, output_path: str):
    """Charger les données"""
    train_data = pd.read_csv(train_data_path)
    test_data = pd.read_csv(test_data_path)
    
    """ Sélection des caractéristiques et de la cible"""
    y = train_data["Survived"]
    features = ["Pclass", "Sex", "SibSp", "Parch"]
    X = pd.get_dummies(train_data[features])
    X_test = pd.get_dummies(test_data[features])
    
    """ Division en ensemble d'entraînement et de validation"""
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=1)
    
    """ Initialisation et entraînement du modèle"""
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
    model.fit(X_train, y_train)
    
    """ Prédictions"""
    predictions = model.predict(X_test)
    
    """ Sauvegarde des prédictions"""
    output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
    output.to_csv(output_path, index=False)
    print("Prédictions sauvegardées avec succès dans", output_path)
    
    # Sauvegarde du modèle
    dump(model, "random_forest_model.joblib")
    print("Modèle sauvegardé avec succès sous random_forest_model.joblib")

# Exemple d'utilisation
if __name__ == "__main__":
    train_and_save_model("data/train.csv", "data/test.csv", "output/submission.csv")
