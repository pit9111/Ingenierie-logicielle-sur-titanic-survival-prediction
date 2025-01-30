import unittest
import pandas as pd
import os
from data_preprocecessing import import_data, analyze_survival  

class TestSurvivalAnalysis(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Charge le fichier train.csv pour les tests."""
        cls.data_path = r"Z:\developpement_logiciel\projet\train.csv" 
        
        if not os.path.exists(cls.data_path):
            raise FileNotFoundError(f"Le fichier {cls.data_path} est introuvable.")
        
        cls.data = import_data(cls.data_path)
    
    def test_import_data(self):
        """Vérifie que les données sont bien importées et non vides."""
        self.assertFalse(self.data.empty, "Le fichier train.csv est vide ou n'a pas été chargé correctement.")
    
    def test_analyze_survival(self):
        """Vérifie que analyze_survival retourne bien des valeurs flottantes sans erreur."""
        try:
            rate_women, rate_men = analyze_survival(self.data)
            self.assertIsInstance(rate_women, float, "Le taux de survie des femmes doit être un float.")
            self.assertIsInstance(rate_men, float, "Le taux de survie des hommes doit être un float.")
        except Exception as e:
            self.fail(f"analyze_survival a levé une exception: {e}")

if __name__ == "__main__":
    unittest.main()
