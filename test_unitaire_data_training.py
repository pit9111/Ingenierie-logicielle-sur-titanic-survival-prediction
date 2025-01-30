import unittest
import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

class TestModelTraining(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        # Définir le chemin des données
        cls.data_path = "Z:\developpement_logiciel\projet"
        cls.model_path = os.path.join(cls.data_path, "trained_model.pkl")
        
        # Charger les fichiers de données
        cls.train_file = os.path.join(cls.data_path, "train.csv")
        cls.test_file = os.path.join(cls.data_path, "test.csv")
        
        if not os.path.exists(cls.train_file) or not os.path.exists(cls.test_file):
            raise FileNotFoundError("Les fichiers train.csv et test.csv doivent exister dans le dossier ../data/")
        
        cls.train_data = pd.read_csv(cls.train_file)
        cls.test_data = pd.read_csv(cls.test_file)
    
    def test_data_loading(self):
        """Test si les fichiers de données sont chargés correctement"""
        self.assertFalse(self.train_data.empty, "Le dataset d'entraînement est vide")
        self.assertFalse(self.test_data.empty, "Le dataset de test est vide")
    
    def test_model_training(self):
        """Test si le modèle est entraîné sans erreur"""
        y = self.train_data["Survived"]
        features = ["Pclass", "Sex", "SibSp", "Parch"]
        X = pd.get_dummies(self.train_data[features])
        X_test = pd.get_dummies(self.test_data[features])
        
        model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
        model.fit(X, y)
        
        # Vérification si le modèle a bien été entraîné
        self.assertIsNotNone(model, "Le modèle n'a pas été entraîné")
        self.assertGreater(len(model.estimators_), 0, "Le modèle n'a pas d'arbres entraînés")
    
    def test_model_saving(self):
        """Test si le modèle est bien sauvegardé"""
        model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
        model.fit(pd.get_dummies(self.train_data[["Pclass", "Sex", "SibSp", "Parch"]]), self.train_data["Survived"])
        
        joblib.dump(model, self.model_path)
        
        self.assertTrue(os.path.exists(self.model_path), "Le fichier du modèle sauvegardé est introuvable")
    
    @classmethod
    def tearDownClass(cls):
        """Nettoyage après les tests"""
        if os.path.exists(cls.model_path):
            os.remove(cls.model_path)

if __name__ == "__main__":
    unittest.main()
