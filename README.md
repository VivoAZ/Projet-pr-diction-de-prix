# Prédiction du prix 

## Description 
Un modèle de régression linéaire pour prédire le prix médian en milliers de dollars d'une maison, basé sur des données historiques des maisons dans la ville de Boston. 

## Fonctionnalité principale 
Prédiction du prix : Ce projet permet de prédire le prix médian des maisons dans la ville de Boston en se basant sur des données historiques. 

## Installation 

### 1- Cloner le dépôt 

git clone https://github.com/VivoAZ/Projet-prediction-de-prix.git

cd Projet-prediction-de-prix 

### 2- Créer et activer un environnement virtuel (venv) 

python -m venv env 

source env/bin/activate  # Pour Linux/macOS 

env\Scripts\activate     # Pour Windows 

### 3- Installer les dépendances 

pip install -r requirements.txt

## Exécution 

Commande pour lancer le projet 
python main.py 

N'oubliez pas de vérifier le chemin d'accès des fichiers main.py et HousingData.csv selon où vous les avez sauvegardés sur votre machine. 

## Structure du projet
main.py : Script principal pour l’entraînement et la prédiction du modèle. 

HousingData.csv : Contient les jeux de données bruts et transformés. 

gradient_boosting_model.pkl : Modèle sauvegardé au format pkl.

Projet_House.ipynb : Notebook Jupyter pour l’analyse exploratoire et les tests. 

requirements.txt : Liste des dépendances nécessaires. 

## Données
Les informations proviennent de la plateforme publique Kaggle.

## Collaboration
Si vous souhaitez contribuer :

1- Forkez le projet. 

2- Créez une branche (git checkout -b ma-branche).

3- Soumettez une Pull Request. 























