import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# Définir le chemin des données
data_path = "../data/"

# Charger les données
train_data = pd.read_csv(os.path.join(data_path, "train.csv"))
test_data = pd.read_csv(os.path.join(data_path, "test.csv"))

# Définir les variables cibles et les features
y = train_data["Survived"]
features = ["Pclass", "Sex", "SibSp", "Parch"]
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

# Entraîner le modèle RandomForest
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)

# Sauvegarder le modèle entraîné dans le dossier data
model_path = os.path.join(data_path, "trained_model.pkl")
joblib.dump(model, model_path)

print(f"Model training completed and saved in '{model_path}'")