# Élements de corrections

## Séance 4

### Questions

- Il manque quelques éléments.

### Code

- Excellent !

## Séance 5

### Questions

- Il manque quelques éléments.

### Code

- Où sont les commentaires demandés dans votre rapport ?

## Séance 6

### Questions

- Il manque quelques éléments.

### Code

- Où sont les commentaires demandés dans votre rapport ?

- Problème d'encodage !

- Il fallait taper `iles['Surface (km²)']`, et non `iles['Surface (km2)']`. Le peu de code rendu fonctionne après cette modification.

## Séance 7

### Questions

- **Question 2.** La rapport de corrélation étudie la liaison entre une variable qualitative et une variable quantitative.

- Il manque quelques éléments. Vous ne développez pas assez.

### Code

- Où sont les commentaires demandés dans votre rapport ?

- Problème d'encodage !

## Séance 8

### Questions

- Il manque quelques éléments. Vous ne développez pas assez.

### Code

- Où sont les commentaires demandés dans votre rapport ?

- Vous avez modifié le code de `def ouvrirUnFichier(nom):` :

```
    def ouvrir_un_fichier(nom):
        """Ouvre un fichier CSV et retourne son contenu sous forme de DataFrame."""
        return pd.read_csv(fichier)
```

mais, pour ouvrir le fichier, il faut taper :
```
    def ouvrir_un_fichier(nom):
        """Ouvre un fichier CSV et retourne son contenu sous forme de DataFrame."""
        with open(nom, "r", encoding='utf-8') as fichier:
            contenu = pd.read_csv(fichier)
        return contenu
```

Pourquoi ? parce que `with` teste l'existence du fichier avant de l'ouvrir.

## Humanités numériques

- Intéressant, mais il fallait développer davantage.

## Remarques générales

- Aucun dépôt régulier sur `GitHub`.

- Merci d'avoir proposé des éléments concrets pour améliorer le cours proposé.

- On sent que vous avez compris. Pourquoi n'avez-vous pas rédigé un tout petit peu plus ?

- Point de détail. Il fallait laisser par dossier le fichier `main.py`. C'est une convention de nommage que vous comprendrez si vous faites du `Python` avancé.

- Attention ! Il ne faut jamais écrire une adresse absolue `"C:\Python\s7\src\data\pib-vs-energie.csv"`, mais toujours une adresse relative `"./src/data/pib-vs-energie.csv"` à partir du dossier racine.

- Excellent travail au niveau du code ! Dommage pour les questions de cours !
