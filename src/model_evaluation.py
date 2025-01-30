import pandas as pd
import joblib
import os

# Définir le chemin des données
data_path = "../data/"

# Charger les données
test_data = pd.read_csv(os.path.join(data_path, "test.csv"))
X_test = pd.get_dummies(test_data[["Pclass", "Sex", "SibSp", "Parch"]])

# Charger le modèle entraîné
model_path = os.path.join(data_path, "trained_model.pkl")
model = joblib.load(model_path)

# Faire des prédictions
predictions = model.predict(X_test)

# Sauvegarder les résultats dans un fichier CSV
output_path = os.path.join(data_path, "submission.csv")
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv(output_path, index=False)

print(f"Predictions saved in '{output_path}'")
