import pytest
import pandas as pd
import sys
import os

# Ajouter le dossier parent au sys.path pour permettre l'import de src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from data_preprocessing import load_data, compute_survival_rate, print_survival_rates

# Charger les données une seule fois pour éviter la duplication
train_data, test_data = load_data()


# ✅ Test 1 : Vérifier que les fichiers CSV sont correctement chargés
def test_load_data():
    assert not train_data.empty, "Le fichier train.csv est vide !"
    assert not test_data.empty, "Le fichier test.csv est vide !"
    assert "Survived" in train_data.columns, "La colonne 'Survived' est absente de train.csv !"


# ✅ Test 2 : Vérifier le calcul du taux de survie des femmes
def test_survival_rate_women():
    rate_women = compute_survival_rate(train_data, "female")
    assert 0 <= rate_women <= 1, "Le taux de survie des femmes doit être entre 0 et 1"


# ✅ Test 3 : Vérifier le calcul du taux de survie des hommes
def test_survival_rate_men():
    rate_men = compute_survival_rate(train_data, "male")
    assert 0 <= rate_men <= 1, "Le taux de survie des hommes doit être entre 0 et 1"


# ✅ Test 4 : Vérifier que la colonne 'Sex' contient bien des valeurs attendues
def test_sex_column():
    assert "Sex" in train_data.columns, "La colonne 'Sex' est absente !"
    unique_values = set(train_data["Sex"].unique())
    assert unique_values.issubset({"male", "female"}), f"Valeurs inattendues dans la colonne 'Sex' : {unique_values}"


# ✅ Test 5 : Vérifier l'affichage de `print_survival_rates()`
def test_print_survival_rates(capfd):
    print_survival_rates(train_data)
    out, err = capfd.readouterr()
    
    assert "of women who survived" in out, "Le texte d'affichage des femmes est absent !"
    assert "of men who survived" in out, "Le texte d'affichage des hommes est absent !"
