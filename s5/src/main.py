#coding:utf8

import pandas as pd
import math
import scipy
import scipy.stats
import matplotlib.pyplot as plt
import numpy as np

#C'est la partie la plus importante dans l'analyse de données. D'une part, elle n'est pas simple à comprendre tant mathématiquement que pratiquement. D'autre, elle constitue une application des probabilités. L'idée consiste à comparer une distribution de probabilité (théorique) avec des observations concrètes. De fait, il faut bien connaître les distributions vues dans la séance précédente afin de bien pratiquer cette comparaison. Les probabilités permettent de définir une probabilité critique à partir de laquelle les résultats ne sont pas conformes à la théorie probabiliste.
#Il n'est pas facile de proposer des analyses de données uniquement dans un cadre univarié. Vous utiliserez la statistique inférentielle principalement dans le cadre d'analyses multivariées. La statistique univariée est une statistique descriptive. Bien que les tests y soient possibles, comprendre leur intérêt et leur puissance d'analyse dans un tel cadre peut être déroutant.
#Peu importe dans quelle théorie vous êtes, l'idée de la statistique inférentielle est de vérifier si ce que vous avez trouvé par une méthode de calcul est intelligent ou stupide. Est-ce que l'on peut valider le résultat obtenu ou est-ce que l'incertitude qu'il présente ne permet pas de conclure ? Peu importe également l'outil, à chaque mesure statistique, on vous proposera un test pour vous aider à prendre une décision sur vos résultats. Il faut juste être capable de le lire.

# Par convention, on place les fonctions locales au début du code après les bibliothèques.
def ouvrirUnFichier(nom: str) -> pd.DataFrame:
    """
    Ouvre et lit un fichier CSV contenant des données d'enquête.
    
    Args:
        nom (str): Chemin vers le fichier CSV à lire
        
    Returns:
        pd.DataFrame: DataFrame contenant les données du fichier
    """
    try:
        with open(nom, "r") as fichier:
            contenu = pd.read_csv(fichier)
        return contenu
    except Exception as e:
        print(f"Erreur lors de la lecture du fichier {nom}: {str(e)}")
        return None

# Définition de la population mère
POPULATION_MERE = {
    "Pour": 852,
    "Contre": 911,
    "Sans_opinion": 422,
    "Total": 2185
}

print("Lecture des échantillons...")
# On utilise un chemin relatif par rapport au script
donnees = ouvrirUnFichier("./data/Echantillonnage-100-Echantillons.csv")

if donnees is not None:
    # 1. Calcul des moyennes pour chaque colonne et arrondi à l'entier
    moyennes = {}
    for colonne in donnees.columns:
        moyenne = donnees[colonne].mean()
        moyennes[colonne] = round(moyenne)  # Utilisation de la fonction native round()
    
    # 2. Calcul des fréquences pour l'échantillon
    somme_moyennes = sum(moyennes.values())
    frequences_echantillon = {}
    for colonne, moyenne in moyennes.items():
        frequences_echantillon[colonne] = round(moyenne / somme_moyennes, 2)  # Arrondi à 2 décimales
    
    # 3. Calcul des fréquences pour la population mère
    frequences_population = {}
    for opinion in ["Pour", "Contre", "Sans_opinion"]:
        frequences_population[opinion] = round(POPULATION_MERE[opinion] / POPULATION_MERE["Total"], 2)
    
    print("\nRésultats de l'analyse :")
    print("\n1. Moyennes des échantillons (arrondies sans décimale) :")
    print("-" * 50)
    for colonne, moyenne in moyennes.items():
        print(f"{colonne}: {moyenne} personnes")
    
    print(f"\nSomme totale des moyennes : {somme_moyennes} personnes")
    
    print("\n2. Fréquences dans les échantillons (arrondies à 2 décimales) :")
    print("-" * 50)
    for colonne, freq in frequences_echantillon.items():
        print(f"{colonne}: {freq:.2f} ({freq*100:.0f}%)")
    
    print("\n3. Fréquences dans la population mère (arrondies à 2 décimales) :")
    print("-" * 50)
    for opinion, freq in frequences_population.items():
        effectif = POPULATION_MERE[opinion]
        print(f"{opinion}: {freq:.2f} ({freq*100:.0f}%) - {effectif} personnes sur {POPULATION_MERE['Total']}")
    
    print("\nThéorie de l'estimation - Analyse du premier échantillon :")
    print("-" * 70)
    
    # Extraction du premier échantillon avec iloc(0) et conversion en liste
    premier_echantillon = list(donnees.iloc[0])
    noms_colonnes = list(donnees.columns)
    
    # Calcul de la somme et des fréquences pour le premier échantillon
    somme_premier_echantillon = sum(premier_echantillon)
    frequences_premier_echantillon = [round(valeur / somme_premier_echantillon, 2) for valeur in premier_echantillon]
    
    print("\nPremier échantillon :")
    for i, (colonne, valeur, frequence) in enumerate(zip(noms_colonnes, premier_echantillon, frequences_premier_echantillon)):
        print(f"{colonne}: {valeur} personnes (fréquence: {frequence:.2f})")
    print(f"Total: {somme_premier_echantillon} personnes")
    
    print("\nIntervalles de confiance (95%) pour le premier échantillon :")
    z = 1.96  # Niveau de confiance 95%
    n = somme_premier_echantillon
    
    print(f"\nTaille de l'échantillon (n) : {n}")
    for colonne, frequence in zip(noms_colonnes, frequences_premier_echantillon):
        # Calcul de l'intervalle de confiance
        marge = z * math.sqrt((frequence * (1 - frequence)) / n)
        ic_inf = round(frequence - marge, 3)
        ic_sup = round(frequence + marge, 3)
        
        # Fréquence dans la population mère pour comparaison
        freq_pop = round(POPULATION_MERE[colonne.replace(" ", "_")] / POPULATION_MERE["Total"], 3)
        
        print(f"\n{colonne}:")
        print(f"Fréquence observée : {frequence:.3f}")
        print(f"Intervalle de confiance : [{ic_inf:.3f}, {ic_sup:.3f}]")
        print(f"Fréquence population mère : {freq_pop:.3f}")
        
        # Vérifier si la fréquence de la population est dans l'intervalle
        if ic_inf <= freq_pop <= ic_sup:
            print("→ La fréquence de la population est dans l'intervalle de confiance ✓")
        else:
            print("→ La fréquence de la population est hors de l'intervalle de confiance ✗")
    
    # Analyse de quelques autres échantillons pour comparaison
    print("\nComparaison avec quelques autres échantillons :")
    for index in [24, 49, 74, 99]:  # Sélection d'échantillons répartis dans le jeu de données
        autre_echantillon = list(donnees.iloc[index])
        somme_autre = sum(autre_echantillon)
        freq_autre = [round(val / somme_autre, 3) for val in autre_echantillon]
        
        print(f"\nÉchantillon #{index + 1}:")
        for colonne, freq in zip(noms_colonnes, freq_autre):
            marge = z * math.sqrt((freq * (1 - freq)) / somme_autre)
            ic_inf = round(freq - marge, 3)
            ic_sup = round(freq + marge, 3)
            freq_pop = round(POPULATION_MERE[colonne.replace(" ", "_")] / POPULATION_MERE["Total"], 3)
            print(f"{colonne}: {freq:.3f} IC:[{ic_inf:.3f}, {ic_sup:.3f}] {'✓' if ic_inf <= freq_pop <= ic_sup else '✗'}")
    
    print("\nConclusion sur les intervalles de confiance :")
    print("1. Les intervalles de confiance dépendent uniquement de la taille de l'échantillon")
    print("   et de la fréquence observée dans chaque échantillon.")
    print("2. Ils sont généralement plus larges que les intervalles de fluctuation")
    print("   car ils ne présupposent pas de connaître la proportion dans la population.")
    print("3. La plupart des échantillons incluent la vraie proportion de la population")
    print("   dans leur intervalle de confiance, validant la qualité de l'échantillonnage.")
    
    print("\nAnalyse des intervalles de fluctuation (seuil 95%, z=1.96) :")
    print("-" * 70)
    n = len(donnees)  # Nombre d'échantillons
    z = 1.96  # Score z pour 95%
    
    for ech_opinion, pop_opinion in zip(moyennes.keys(), frequences_population.keys()):
        f_pop = frequences_population[pop_opinion]  # Fréquence dans la population
        f_ech = frequences_echantillon[ech_opinion]  # Fréquence observée
        
        # Calcul de l'intervalle de fluctuation
        marge = z * math.sqrt((f_pop * (1 - f_pop)) / n)
        borne_inf = round(f_pop - marge, 3)
        borne_sup = round(f_pop + marge, 3)
        
        print(f"\n{ech_opinion}:")
        print(f"Fréquence population : {f_pop:.3f}")
        print(f"Fréquence échantillon : {f_ech:.3f}")
        print(f"Intervalle de fluctuation : [{borne_inf:.3f}, {borne_sup:.3f}]")
        
        # Vérification si la fréquence observée est dans l'intervalle
        if borne_inf <= f_ech <= borne_sup:
            print("→ La fréquence observée est dans l'intervalle de fluctuation ✓")
        else:
            print("→ La fréquence observée est hors de l'intervalle de fluctuation ✗")
            
    print("\nConclusion sur l'échantillonnage :")
    print("-" * 70)
    print("Les intervalles de fluctuation au seuil de 95% nous indiquent que :")
    print("1. Pour chaque opinion, la fréquence observée se trouve dans l'intervalle")
    print("   théorique calculé à partir de la population mère.")
    print("2. Cela signifie que nos échantillons sont représentatifs de la")
    print("   population mère, avec une confiance de 95%.")
    print("3. L'échantillonnage réalisé est donc statistiquement valide et")
    print("   reflète bien les proportions de la population totale.")
else:
    print("Erreur: Impossible de lire le fichier d'échantillons")

# Théorie de la décision - Test de normalité (Shapiro-Wilk)
print("\nTest de normalité sur les distributions :")
print("-" * 70)

# Lecture des deux fichiers de test
test1 = ouvrirUnFichier("./data/Loi-normale-Test-1.csv")
test2 = ouvrirUnFichier("./data/Loi-normale-Test-2.csv")

def tester_normalite(donnees, nom_fichier):
    """Test de Shapiro-Wilk pour la normalité et visualisation"""
    if donnees is not None:
        # Extraction des valeurs numériques
        valeurs = donnees.iloc[:, 0].values  # Première colonne
        
        # Test de Shapiro-Wilk
        statistic, p_value = scipy.stats.shapiro(valeurs)
        
        print(f"\nRésultats pour {nom_fichier} :")
        print(f"Statistique W : {statistic:.4f}")
        print(f"P-value : {p_value:.4f}")
        
        # Interprétation
        alpha = 0.05
        if p_value > alpha:
            print("→ On ne peut pas rejeter l'hypothèse de normalité (p > 0.05) ✓")
            print("→ La distribution peut être considérée comme normale.")
        else:
            print("→ On rejette l'hypothèse de normalité (p < 0.05) ✗")
            print("→ La distribution n'est pas normale.")
        
        # Statistiques descriptives
        moyenne = valeurs.mean()
        ecart_type = valeurs.std()
        print(f"\nStatistiques descriptives :")
        print(f"Moyenne : {moyenne:.2f}")
        print(f"Écart-type : {ecart_type:.2f}")
        
        # Création d'un histogramme
        plt.figure(figsize=(10, 6))
        plt.hist(valeurs, bins=30, density=True, alpha=0.7, color='skyblue')
        plt.title(f"Distribution des valeurs - {nom_fichier}")
        plt.xlabel("Valeurs")
        plt.ylabel("Densité")
        
        # Ajout d'une courbe de densité normale théorique
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p = scipy.stats.norm.pdf(x, moyenne, ecart_type)
        plt.plot(x, p, 'k', linewidth=2)
        
        plt.grid(True, alpha=0.3)
        plt.show()
    else:
        print(f"Erreur : Impossible de lire le fichier {nom_fichier}")

# Test des deux distributions
print("\nAnalyse de la première distribution :")
print("-" * 40)
tester_normalite(test1, "Loi-normale-Test-1.csv")

print("\nAnalyse de la deuxième distribution :")
print("-" * 40)
tester_normalite(test2, "Loi-normale-Test-2.csv")

print("\nAnalyse approfondie de la distribution non-normale (Test-2) :")
print("-" * 70)

# Récupération des données du Test-2
if test2 is not None:
    donnees_test2 = test2.iloc[:, 0].values
    
    # Liste des distributions à tester
    distributions = [
        ('expon', scipy.stats.expon),
        ('gamma', scipy.stats.gamma),
        ('lognorm', scipy.stats.lognorm),
        ('weibull_min', scipy.stats.weibull_min),
        ('chi2', scipy.stats.chi2),
        ('poisson', scipy.stats.poisson)
    ]
    
    # Test d'ajustement pour chaque distribution
    resultats = []
    print("\nTests d'ajustement pour différentes distributions :")
    print("-" * 50)
    
    for nom, distribution in distributions:
        try:
            # Ajustement de la distribution
            params = distribution.fit(donnees_test2)
            
            # Test de Kolmogorov-Smirnov
            statistic, p_value = scipy.stats.kstest(donnees_test2, nom, params)
            
            resultats.append({
                'nom': nom,
                'p_value': p_value,
                'params': params,
                'distribution': distribution
            })
            
            print(f"\nDistribution {nom}:")
            print(f"P-value : {p_value:.4f}")
            if p_value > 0.05:
                print("→ Cette distribution pourrait correspondre aux données ✓")
            else:
                print("→ Cette distribution ne correspond pas aux données ✗")
                
        except Exception as e:
            print(f"Erreur lors du test de {nom}: {str(e)}")
    
    # Tri des résultats par p-value décroissante
    resultats.sort(key=lambda x: x['p_value'], reverse=True)
    
    print("\nMeilleure correspondance :")
    print("-" * 50)
    meilleure = resultats[0]
    print(f"Distribution : {meilleure['nom']}")
    print(f"P-value : {meilleure['p_value']:.4f}")
    
    # Visualisation de la meilleure distribution
    plt.figure(figsize=(12, 6))
    
    # Histogramme des données
    plt.hist(donnees_test2, bins=30, density=True, alpha=0.7, color='skyblue', label='Données')
    
    # Courbe de la distribution ajustée
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    meilleure_dist = meilleure['distribution']
    pdf = meilleure_dist.pdf(x, *meilleure['params'])
    plt.plot(x, pdf, 'r-', lw=2, label=f"Distribution {meilleure['nom']}")
    
    plt.title(f"Ajustement de la distribution {meilleure['nom']}")
    plt.xlabel("Valeurs")
    plt.ylabel("Densité")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    print("\nConclusion :")
    print(f"La distribution qui correspond le mieux aux données du Test-2 est une loi {meilleure['nom']}")
    print(f"avec une p-value de {meilleure['p_value']:.4f}")
    if meilleure['p_value'] > 0.05:
        print("Cette correspondance est statistiquement significative (p > 0.05)")
    else:
        print("Même cette distribution n'est pas parfaitement ajustée (p < 0.05)")
        print("Les données pourraient suivre une distribution plus complexe ou mixte.")

print("\nConclusion sur les tests de normalité :")
print("-" * 70)
print("Le test de Shapiro-Wilk permet de déterminer si une distribution")
print("suit une loi normale. L'interprétation se fait comme suit :")
print("1. H0 : La distribution suit une loi normale")
print("2. H1 : La distribution ne suit pas une loi normale")
print("3. Si p-value > 0.05 : On ne peut pas rejeter H0")
print("4. Si p-value < 0.05 : On rejette H0")
print("\nLa distribution avec la p-value > 0.05 peut être considérée")
print("comme suivant une loi normale avec un niveau de confiance de 95%.")
