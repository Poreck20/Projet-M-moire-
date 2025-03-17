
import streamlit as st
import numpy as np

# Exemple de fonction de prédiction (remplacer avec ta vraie fonction)
def predire_prix(adresse, surface_bati, surface_terrain, nombre_pieces, code_departement, type_bien):
    return 200000  # Exemple de prix, remplace-le par ta vraie logique

st.title("Estimation du prix immobilier")

# Saisie des informations par l'utilisateur
adresse = st.text_input("Entrez l'adresse du bien immobilier")
surface_bati = st.number_input("Surface du bâtiment (en m²)", min_value=1, step=1)
surface_terrain = st.number_input("Surface du terrain (en m²)", min_value=1, step=1)
nombre_pieces = st.number_input("Nombre de pièces", min_value=1, step=1)
code_departement = st.selectbox("Code département", ["75", "77", "78", "91", "92", "93", "94", "95"])
type_bien = st.selectbox("Type de bien", ["Maison", "Appartement"])

# Bouton pour calculer le prix
if st.button("Calculer le prix estimé"):
    # Vérifier si toutes les informations nécessaires sont présentes
    if adresse and surface_bati > 0 and surface_terrain > 0 and nombre_pieces > 0:
        # Appel à la fonction de prédiction
        prix_estime = predire_prix(adresse, surface_bati, surface_terrain, nombre_pieces, code_departement, type_bien)
        st.write(f"Le prix estimé du bien est : {prix_estime} €")
    else:
        st.error("Veuillez remplir toutes les informations nécessaires.")
