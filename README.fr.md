# Session de TP : Création d'un système de recommandation avec LightFM

## Présentation

Au cours de cette session de TP, nous passerons en revue l'ensemble du processus de création d'un système de recommandation. Nous utiliserons le [jeu de données H&M publié dans le cadre d'un concours Kaggle](https://www.kaggle.com/c/h-and-m-personalized-fashion-recommendations) et la [bibliothèque LightFM](https://making.lyst.com/lightfm/docs/home.html). La session couvrira l'analyse des données, l'échantillonnage des données, l'entraînement du modèle, le réglage des hyperparamètres, l'évaluation et la recommandation hybride intégrant les caractéristiques des articles.

**Présentation du projet** : [Disponible sur Google Drive](https://drive.google.com/drive/folders/1Y7SJnwZp1KZxfYF64PqIM8drlQqJKezw)

## Ensemble de données

Téléchargez l'[ensemble de données H&M depuis la page du concours Kaggle](https://www.kaggle.com/c/h-and-m-personalized-fashion-recommendations) ou utilisez les données disponibles dans le dossier gdrive.

### Fichiers nécessaires

- transactions_train.csv
- articles.csv
- customers.csv

## Introduction à LightFM

### Qu'est-ce que LightFM ?

LightFM est une bibliothèque Python conçue pour créer et évaluer des systèmes de recommandation. Elle est particulièrement adaptée aux scénarios de recommandation hybrides qui combinent le filtrage collaboratif et les méthodes basées sur le contenu. LightFM est réputée pour sa flexibilité, qui vous permet d'intégrer les métadonnées des utilisateurs et des éléments dans le processus de recommandation, ce qui peut améliorer considérablement la précision de vos recommandations.

### Principales caractéristiques de LightFM

1. Modèles hybrides flexibles : LightFM permet de combiner le filtrage collaboratif et le filtrage basé sur le contenu en intégrant les caractéristiques des éléments et des utilisateurs.
2. Différentes fonctions de perte : LightFM prend en charge plusieurs fonctions de perte pour l'entraînement des modèles, notamment :
- WARP (Weighted Approximate-Rank Pairwise) : optimise la qualité du classement, convient aux données de rétroaction implicite.
    - BPR (Bayesian Personalized Ranking) : optimise le classement par paires, couramment utilisé dans les scénarios de feedback implicite.
- Logistique : convient au feedback explicite.
- WARP-kos : une variante de WARP à utiliser avec des ensembles de données très clairsemés.
3.    Évolutivité : conçu pour traiter efficacement de grands ensembles de données.
4.    Facilité d'utilisation : fournit une API simple et intuitive pour l'entraînement et l'évaluation des modèles.

### Composants d'un modèle LightFM

1.    Matrice d'interactions : représente les interactions entre les utilisateurs et les articles. Dans notre cas, il s'agira d'une matrice creuse où les lignes représentent les utilisateurs, les colonnes les articles et les valeurs les interactions (par exemple, les achats).
2.    Caractéristiques des utilisateurs et des articles : matrices facultatives qui contiennent des informations supplémentaires sur les utilisateurs et les articles. Pour cet exercice, nous intégrerons les caractéristiques des articles afin de créer un modèle hybride.
3.    Fonction de perte : définit la manière dont le modèle est entraîné. Nous testerons différentes fonctions de perte afin d'optimiser nos recommandations.

## Guide étape par étape

## Étape 1 : Exploration et compréhension des données

Objectif : se familiariser avec la structure et les caractéristiques de l'ensemble de données H&M.

Questions clés à explorer :
- À quoi ressemblent les données d'interaction ? Combien d'utilisateurs et d'articles uniques avons-nous ?
- Quelle est la rareté de l'ensemble de données ? (Comparez le nombre total d'interactions possibles par rapport au nombre d'interactions réelles)
- Comment les interactions sont-elles réparties entre les utilisateurs et les articles ? Y a-t-il des utilisateurs intensifs ou des articles à succès ?
- Quelle période les données couvrent-elles ? Y a-t-il des tendances saisonnières ?
- Quelles métadonnées sont disponibles pour les articles et les clients ?

Analyses suggérées :
- Tracer la distribution des interactions par utilisateur et par article (histogrammes, boîtes à moustaches)
- Identifier la longue traîne : quel pourcentage d'articles/d'utilisateurs représente 80 % des interactions ?
- Examiner les articles les plus et les moins populaires - quelles tendances remarquez-vous ?
- Réfléchissez à la question suivante : comment ces tendances pourraient-elles influencer votre stratégie de recommandation ?

## Étape 2 : Stratégie d'échantillonnage des données

Objectif : créer un ensemble de données gérable pour l'expérimentation tout en préservant les caractéristiques importantes.

Pourquoi échantillonner ? Les ensembles de données complets peuvent être coûteux en termes de calcul pour l'expérimentation. Un échantillonnage intelligent vous aide à itérer rapidement.

Considérations relatives à l'échantillonnage :
- Faut-il échantillonner les utilisateurs, les éléments ou les interactions ? Quels sont les compromis ?
- Comment pouvez-vous conserver les caractéristiques de distribution des données d'origine ?
- Envisagez différentes stratégies d'échantillonnage : aléatoire, stratifié ou basé sur les niveaux d'activité
- Recommandation : commencez par les utilisateurs actifs (par exemple, les utilisateurs ayant plus de 5 interactions) et les éléments populaires

Expérimentation : essayez différentes tailles d'échantillon (1 000, 10 000, 50 000 interactions) et observez leur impact sur les performances du modèle.

## Étape 3 : Prétraitement des données et construction de la matrice 

Objectif : Transformer les données brutes en formats adaptés à LightFM.

Tâches clefs :

- Mappage des identifiants : créer des mappages entiers pour les identifiants des utilisateurs et des éléments (LightFM nécessite des indices entiers)
- Matrice d'interaction : construire une matrice utilisateur-article clairsemée
- Plus d'informations [ici](https://making.lyst.com/lightfm/docs/examples/dataset.html), dans les sections « Building the ID mappings » (Construire les mappages d'ID) et « Building the interactions matrix » (Construire la matrice d'interactions).
- Nettoyage des données : traiter les doublons, les valeurs aberrantes ou les entrées non valides

## Étape 4 : Stratégie de répartition entre entraînement et test

Objectif : créer une configuration d'évaluation robuste qui simule des scénarios réels.
Documentation LightFM : [Validation croisée](https://making.lyst.com/lightfm/docs/cross_validation.html)

Stratégies de division à envisager :
- Division temporelle : utiliser des divisions basées sur le temps (plus réalistes pour les systèmes de recommandation)
- Division aléatoire : diviser les interactions de manière aléatoire pour chaque utilisateur
- Division basée sur l'utilisateur : exclure complètement certains utilisateurs pour les tests

## Étape 5 : Formation du modèle

Objectif : construire et comprendre un modèle de filtrage collaboratif de base.
Documentation LightFM : [classe de modèle](https://making.lyst.com/lightfm/docs/lightfm.html)

Configuration expérimentale :
- Commencez par un modèle simple utilisant uniquement les données d'interaction (sans caractéristiques)
- Essayez différentes fonctions de perte : WARP, BPR, logistique - laquelle fonctionne le mieux pour vos données ?
- Expérimentez avec différents nombres de facteurs latents (dimensions)

Paramètres à explorer :
- no_components : commencez avec 30-50, essayez avec plus
- loss : commencez avec « warp » pour le feedback implicite
- learning_rate : essayez des valeurs comprises entre 0,01 et 0,1
- epochs : surveillez la convergence (commencez avec 10-20)

## Étape 6 : Optimisation des hyperparamètres

Objectif : Améliorer systématiquement les performances du modèle.

Options d'approche :
- Recherche manuelle par grille : essayer différentes combinaisons de paramètres clés.
- Recherche aléatoire : plus efficace que la recherche par grille pour de nombreux paramètres.
- Basée sur la validation : utiliser un ensemble de validation distinct ou une validation croisée.

Paramètres clés à ajuster :
- Nombre de facteurs latents.
- Taux d'apprentissage et régularisation.
- Choix de la fonction de perte.
- Nombre d'époques (surveiller le surajustement).

## Étape 7 : Évaluation et interprétation du modèle

Objectif : Évaluer la qualité du modèle à l'aide de mesures appropriées.
Documentation LightFM : [Évaluation](https://making.lyst.com/lightfm/docs/lightfm.evaluation.html)

Mesures à prendre en compte :
- Precision@K & Recall@K : Combien d'éléments pertinents dans les recommandations top-K ?
- AUC : qualité globale du classement
- NDCG : prend en compte l'ordre de classement des éléments pertinents

Questions d'évaluation :
- Comment les performances varient-elles en fonction de K (top 5 vs top 20) ?
- Y a-t-il des différences de performances entre les segments d'utilisateurs (utilisateurs actifs vs occasionnels) ?
- À quoi ressemble une « bonne » performance pour votre cas d'utilisation ?

Au-delà des chiffres :
- Inspectez manuellement les recommandations pour quelques utilisateurs : sont-elles intuitives ?
- Vérifiez la diversité : recommandez-vous uniquement des éléments populaires ?

## Étape 8 : Modèle hybride avec caractéristiques des articles

Objectif : intégrer les métadonnées des articles afin d'améliorer les recommandations, en particulier pour les articles à démarrage à froid.
Créer des caractéristiques d'articles (basées sur la [classe de données](https://making.lyst.com/lightfm/docs/examples/dataset.html)) et former un modèle hybride

Expériences sur le modèle hybride :
- Comparer les performances du filtrage collaboratif pur et celles du modèle hybride
- Tester des scénarios de démarrage à froid : dans quelle mesure le modèle recommande-t-il de nouveaux articles ?
- Ablation des caractéristiques : quelles caractéristiques contribuent le plus aux performances ?

Aléatoire et popularité + 2 versions de LightFM
2 catégories d'approches :
- 1ère version : filtrage collaboratif
- 2ème version : approche hybride = identifiant utilisateur, identifiant article, caractéristiques des articles

## Environnement de développement & Structure du projet

### Création d'un environnement virtuel (en ligne de commande)

```bash
python -m venv venv && source venv/bin/activate && pip install -r requirements-dev.txt
```

### Arborescence

.
├── README.md                           ← Documentation principale (EN)
├── README.fr.md                        ← Version française
├── requirements.txt                    ← Dépendances minimales
├── requirements-dev.txt                ← Dépendances de développement
├── venv/                               ← Environnement virtuel
├── notebooks/
│   └── LightFM_HM_recommender.ipynb    ← notebook de travail
├── data/                               ← fichiers de données brutes
│   └── raw/
│       ├── articles.csv
│       ├── customers.csv
│       └── transactions_train.csv