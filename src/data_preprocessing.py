import os
import pandas as pd

def load_data():
    """
    Charge les fichiers CSV des données d'entraînement et de test.

    :return: DataFrame train et test
    """
    # Obtenir le chemin du dossier parent (racine du projet)
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    train_path = os.path.join(BASE_DIR, "data", "train.csv")
    test_path = os.path.join(BASE_DIR, "data", "test.csv")

    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    
    return train_data, test_data


def compute_survival_rate(data: pd.DataFrame, gender: str) -> float:
    """
    Calcule le taux de survie d'un genre spécifique.

    :param data: DataFrame contenant les données des passagers
    :param gender: "male" ou "female"
    :return: Taux de survie pour le genre donné
    """
    subset = data.loc[data.Sex == gender]["Survived"]
    return sum(subset) / len(subset) if len(subset) > 0 else 0


def print_survival_rates(train_data: pd.DataFrame):
    """
    Affiche les taux de survie pour les hommes et les femmes.

    :param train_data: DataFrame des données d'entraînement
    """
    rate_women = compute_survival_rate(train_data, "female")
    rate_men = compute_survival_rate(train_data, "male")

    print(f"% of women who survived: {rate_women:.2%}")
    print(f"% of men who survived: {rate_men:.2%}")


if __name__ == "__main__":
    # Charger les données avec la nouvelle fonction
    train_data, test_data = load_data()

    # Afficher les taux de survie
    print_survival_rates(train_data)
