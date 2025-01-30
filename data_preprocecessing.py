import pandas as pd


def import_data(filepath):
    """Charge les données depuis un fichier CSV."""
    return pd.read_csv(filepath)

def analyze_survival(data):
    """Analyse et affiche les taux de survie des hommes et des femmes."""
    women = data.loc[data.Sex == 'female']["Survived"]
    rate_women = sum(women) / len(women)
    print("% of women who survived:", rate_women)
    
    men = data.loc[data.Sex == 'male']["Survived"]
    rate_men = sum(men) / len(men)
    print("% of men who survived:", rate_men)

# Exemple d'utilisation
data_path = "Z:\developpement_logiciel\projet"
train_data = import_data(data_path)
analyze_survival(train_data)


