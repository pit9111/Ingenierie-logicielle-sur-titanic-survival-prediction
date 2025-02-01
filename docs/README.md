# Ingenierie logicielle sur titanic survival prediction
Project Objectives

The objective of this project is to predict the survival of Titanic passengers using data analysis and machine learning techniques. With the available data, we will identify factors influencing survival and train classification models to predict passenger survival rates.

Project Structure
```sh
TITANIC/
├── .github/
│   └── workflows/
│       ├── ci.yml  # GitHub Actions pour CI/CD
├── data/
│   ├── gender_submission.csv
│   ├── test.csv
│   └── train.csv
├── src/
│   ├── __init__.py  # Fichier permettant l'import en tant que package
│   ├── data_preprocessing.py
│   ├── model_evaluation.py
│   ├── model_training.py
├── tests/
│   ├── test_data_preprocessing.py
│   ├── test_model_evaluation.py
│   ├── test_model_training.py
├── .gitignore  # Ignore les fichiers inutiles pour Git
├── README.md
├── requirements.txt  # Liste des dépendances du projet
├── titanic-tutorial.ipynb  # Notebook Jupyter pour analyse exploratoire


```
##  Getting Started

**System Requirements:**

* **Python**: `version 3.`

###  Installation

<h4>From <code>source</code></h4>

> 1. Clone the  repository:
>
> ```console
> git clone ../
> ```
>
> 2. Change to the project directory:
> ```console
> cd ./Ingenierie-logicielle-sur-titanic-survival-prediction
> ```
>
> 3. Create a virtualenv:
> ```console
> python -m venv env
> ```
>


>
> 4. Launch the virtualenv
> ```console
> env\Scripts\activate


> 5. Install the dependencies:
> ```console
> pip install -r requirements.txt
>
> ```
>
