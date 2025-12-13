import pandas as pd
import math
import matplotlib.pyplot as plt

# Charger le fichier CSV
df = pd.read_csv("C:\Python\s6\src\data\island-index.csv")

# Isoler la colonne « Surface (km²) » et convertir en float
surface = list(df["Surface (km²)"].dropna().astype(float))

# Ajouter les continents
surface.extend([
    85545323,   # Asie/Afrique/Europe
    37856841,   # Amérique
    7768030,    # Antarctique
    7605049     # Australie
])

# Fonction pour trier une liste en ordre décroissant
def ordreDecroissant(liste):
    return sorted(liste, reverse=True)

# Fonction pour appliquer le logarithme à chaque élément d'une liste
def conversionLog(liste):
    return [math.log(element) for element in liste]

# Ordonner la liste
surface_ordonnee = ordreDecroissant(surface)

# Calculer les logs des surfaces et des rangs
log_surface = conversionLog(surface_ordonnee)
log_rang = conversionLog(list(range(1, len(surface_ordonnee) + 1)))

# Visualiser la loi rang-taille
plt.figure(figsize=(8, 6))
plt.plot(log_rang, log_surface, marker='o')
plt.xlabel("log(Rang)")
plt.ylabel("log(Surface (km²))")
plt.title("Loi rang-taille des surfaces")
plt.grid(True)
plt.show()


# Oui, il est possible de faire un test statistique sur les rangs.
# Par exemple, on peut tester si la distribution des rangs suit une loi particulière (loi de Zipf, loi de puissance, etc.)
# ou vérifier la corrélation entre log(rang) et log(surface) avec un test de corrélation (Pearson, Spearman).
# Exemple de test de corrélation de Spearman :

# from scipy.stats import spearmanr

# Test de corrélation de Spearman entre log(rang) et log(surface)
# corr, p_value = spearmanr(log_rang, log_surface)
# print(f"Spearman correlation: {corr:.3f}, p-value: {p_value:.3g}")

# Si la corrélation est proche de -1 et la p-value faible, cela confirme une forte relation rang-taille.

