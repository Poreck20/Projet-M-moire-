{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "ada9f6ef-0286-4229-a70c-5edc5737f6f0",
   "metadata": {},
   "source": [
    "# Projet : Analyse et Modélisation de la Valeur Foncière \n",
    "### Etudiant : Diallo Alpha M1 Maserati & GPIA "
   ]
  },
  {
   "cell_type": "markdown",
   "id": "9dfa5187-6d19-4784-a78a-5b1dce9a3336",
   "metadata": {},
   "source": [
    "Ce projet vise à analyser et modéliser la valeur foncière des biens immobiliers en utilisant des techniques de régression et d’apprentissage automatique. L'objectif est de comprendre les facteurs influençant le prix au m² des biens immobiliers en Île-de-France à partir des données de **data.gouv.fr**.  "
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b17a4e07-b5b2-4b38-9eba-254389de5247",
   "metadata": {},
   "source": [
    "## 📂 Contenu du projet"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e3b5e6bc-a87d-4892-ace5-43e6f92d85d1",
   "metadata": {},
   "source": [
    "- **`main.py`** : Contient les scripts principaux pour le traitement des données et la modélisation.  \n",
    "- **`Memoire.ipynb`** : Un notebook détaillant les analyses exploratoires, les prétraitements des données et les résultats des modèles.  "
   ]
  },
  {
   "cell_type": "markdown",
   "id": "eafbd1eb-98f9-471d-ae02-34787ab8740a",
   "metadata": {},
   "source": [
    "## 🔍 Méthodologie  "
   ]
  },
  {
   "cell_type": "markdown",
   "id": "5d0ba6b6-8d03-4211-b9e5-9f68932c2499",
   "metadata": {},
   "source": [
    "1. **Exploration et nettoyage des données**  \n",
    "   - Gestion des valeurs aberrantes (ex. nombre de pièces, surface incohérente, etc.).  \n",
    "   - Traitement des valeurs manquantes.  \n",
    "   - Transformation des variables (logarithme, encodage des variables catégorielles)."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "c37d87fe-28d6-48d3-825b-de76a833e723",
   "metadata": {},
   "source": [
    "2. **Modélisation**  \n",
    "   - Régression linéaire (OLS) pour une première analyse des relations entre variables.  \n",
    "   - Random Forest Regressor pour améliorer les performances prédictives.  \n",
    "   - Évaluation des performances via les métriques **R², RMSE, MAE**. "
   ]
  },
  {
   "cell_type": "markdown",
   "id": "7d727097-d870-498c-9d85-afc195611a49",
   "metadata": {},
   "source": [
    "3. **Analyse des résultats**  \n",
    "   - Importance des variables dans le modèle.  \n",
    "   - Visualisation des résidus et de la distribution des prix.  "
   ]
  },
  {
   "cell_type": "markdown",
   "id": "7e432d25-3537-4c57-b6f2-139d0de9257a",
   "metadata": {},
   "source": [
    "## 🚀 Installation et Exécution"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "c6ac7798-45e0-47fe-8a81-e7a834d5b7d1",
   "metadata": {},
   "source": [
    "```bash\n",
    "git clone https://github.com/ton-projet.git\n",
    "cd ton-projet"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
