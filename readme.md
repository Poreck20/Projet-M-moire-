# Projet : Analyse et Modélisation de la Valeur Foncière 
### Etudiant : Diallo Alpha M1 Maserati & GPIA 

Ce projet vise à analyser et modéliser la valeur foncière des biens immobiliers en utilisant des techniques de régression et d’apprentissage automatique. L'objectif est de comprendre les facteurs influençant le prix au m² des biens immobiliers en Île-de-France à partir des données de **data.gouv.fr**.  

## Contenu du projet

- **`main.py`** : Contient les scripts principaux pour le traitement des données et la modélisation.  
- **`Memoire.ipynb`** : Un notebook détaillant les analyses exploratoires, les prétraitements des données et les résultats des modèles.  

## 🔍 Méthodologie  

1. **Exploration et nettoyage des données**  
   - Gestion des valeurs aberrantes (ex. nombre de pièces, surface incohérente, etc.).  
   - Traitement des valeurs manquantes.  
   - Transformation des variables (logarithme, encodage des variables catégorielles).

2. **Modélisation**  
   - Régression linéaire (OLS) pour une première analyse des relations entre variables.  
   - Random Forest Regressor pour améliorer les performances prédictives.  
   - Évaluation des performances via les métriques **R², RMSE, MAE**. 

3. **Analyse des résultats**  
   - Importance des variables dans le modèle.  
   - Visualisation des résidus et de la distribution des prix.  

## 🚀 Installation et Exécution

```bash
git clone https://github.com/ton-projet.git
cd ton-projet
