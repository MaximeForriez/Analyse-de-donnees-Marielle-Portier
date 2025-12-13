# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import os

# Vérifie que le fichier existe avant de le lire
chemin_fichier = os.path.join("data", "resultats-elections-presidentielles-2022-1er-tour.csv")

if not os.path.exists(chemin_fichier):
    print(f"⚠️ Erreur : le fichier {chemin_fichier} est introuvable.")
else:
    print("✅ Fichier trouvé, lecture en cours...\n")

    # Lecture du CSV avec Pandas (inutile d’utiliser 'with open' ici)
    contenu = pd.read_csv(chemin_fichier, encoding="utf-8")

    # Affiche les 5 premières lignes pour vérifier la lecture
    print("=== Aperçu du fichier ===")
    print(contenu.head())

    # --- Nombre de lignes et colonnes ---
    nb_lignes = len(contenu)
    nb_colonnes = len(contenu.columns)

    print(f"\nNombre de lignes : {nb_lignes}")
    print(f"Nombre de colonnes : {nb_colonnes}")

    # --- Types des colonnes ---
    print("\n=== Types détectés par pandas ===")
    print(contenu.dtypes)

    # --- Conversion des types pandas vers types simples ---
    def map_dtype(dtype):
        dtype_str = str(dtype)
        if "int" in dtype_str:
            return "int"
        elif "float" in dtype_str:
            return "float"
        elif "bool" in dtype_str:
            return "bool"
        else:
            return "str"

    types_simple = {col: map_dtype(typ) for col, typ in contenu.dtypes.items()}

    print("\n=== Types simplifiés ===")
    for col, typ in types_simple.items():
        print(f"{col} : {typ}")

    # --- Affiche uniquement la première ligne du tableau ---
    print("\n=== Première ligne du tableau ===")
    print(contenu.head(1))
    



# Mettre dans un commentaire le numéro de la question
# Question 1
# ...
