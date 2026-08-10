# Suivi du projet bois-énergie

Dernière mise à jour : 10 août 2026

## Objectif

Développer deux modèles permettant d'estimer et de prévoir, par région et par année, l'offre et la demande en bois-énergie à Madagascar, puis mesurer leur écart.

## Choix de modélisation

Le ML classique a été retenu comme approche principale. Les données ont une structure de panel région-année et un volume modéré : environ 22 régions observées de 2000 à 2030. Ce volume est adapté aux régressions régularisées, Random Forest et Gradient Boosting, mais insuffisant pour justifier des réseaux de neurones profonds.

Les modèles de séries temporelles indépendants par région disposeraient de trop peu d'observations. La dimension temporelle sera donc intégrée au ML classique avec des retards, moyennes mobiles, variations annuelles et une validation chronologique.

Une régression linéaire ou ElasticNet servira de référence interprétable. Random Forest et Gradient Boosting seront ensuite comparés. Le choix final reposera sur les performances hors période d'entraînement, la stabilité et l'interprétabilité.

## Principes méthodologiques

- Les fichiers de `data/raw/` ne sont jamais modifiés.
- Les données historiques et les projections de scénario sont séparées après nettoyage.
- La séparation entraînement-validation-test respecte l'ordre temporel.
- Les projections ne sont jamais utilisées pour entraîner ou évaluer le modèle historique.
- Les valeurs manquantes ne sont pas imputées pendant le nettoyage. L'imputation sera ajustée uniquement sur l'ensemble d'entraînement pour éviter les fuites de données.
- Les totaux sont recalculés uniquement lorsque leurs deux composantes sont disponibles.
- Chaque transformation doit être reproductible par un script et vérifiée par des tests.
- Les données actuelles sont synthétiques : elles permettent de valider la méthode, pas encore de produire des estimations officielles.

## Avancement

### 3 août 2026 — Structure et versionnement

- Organisation du dépôt en dossiers `data`, `src`, `models`, `reports`, `notebooks`, `tests` et `docs`.
- Configuration de `.gitignore` pour exclure données privées, sorties générées, modèles entraînés, caches et secrets.
- Conservation des CSV synthétiques comme jeux de développement reproductibles.

### 3 août 2026 — Nettoyage et harmonisation

- Ajout d'un pipeline commun pour l'offre et la demande dans `src/data/clean_data.py`.
- Lecture correcte des CSV séparés par un point-virgule.
- Suppression des espaces parasites et harmonisation des valeurs textuelles.
- Conversion des nombres contenant des espaces de milliers ou une virgule décimale.
- Réparation des années mal formées à partir de l'identifiant de l'observation.
- Harmonisation des 22 noms de régions, notamment `sava`, `DIANA ` et `Amoron i Mania`.
- Conversion des marqueurs `ND`, `-` et champs vides en valeurs manquantes.
- Remplacement par une valeur manquante des nombres négatifs impossibles et des pourcentages hors de l'intervalle 0-100.
- Recalcul des totaux offre/demande lorsque les deux composantes sont disponibles.
- Suppression des doublons exacts et contrôle de l'unicité région-année.
- Séparation des données historiques et des projections.
- Génération de rapports qualité JSON et ajout de tests automatisés.

Résultat de l'exécution : chaque fichier passe de 694 à 682 lignes après suppression de 12 doublons. Chaque panel contient 22 régions sans doublon région-année. Les données historiques comptent 550 lignes de 2000 à 2024 et les projections 132 lignes de 2025 à 2030.

Les valeurs manquantes restantes sont conservées et comptabilisées dans `reports/generated/qualite_demande.json` et `reports/generated/qualite_offre.json`. Elles seront traitées dans le futur pipeline de préparation ML afin que l'imputation soit apprise seulement sur les données d'entraînement.

### 4 août 2026 — Analyse exploratoire

- Ajout du module reproductible `src/analysis/explore_data.py`.
- Analyse limitée aux 550 observations historiques de chaque jeu, couvrant 22 régions de 2000 à 2024.
- Génération des statistiques descriptives, taux de valeurs manquantes, candidats aux valeurs extrêmes selon la règle IQR et matrices de corrélation.
- Génération de cinq graphiques par jeu : données manquantes, distributions des cibles, tendance nationale, évolution région-année et corrélations.
- Production de rapports JSON, CSV et Markdown dans `reports/generated/`.
- Ajout de tests automatisés pour les statistiques et la description du panel temporel.

Premiers constats : les variables métier ont au maximum 2,182 % de valeurs manquantes. La demande totale est fortement corrélée à la population (0,9415) et au nombre de ménages (0,8905). L'offre totale est fortement corrélée au volume transporté (0,9469). Les volumes transportés et les productions converties devront être examinés avant modélisation, car une variable calculée à partir de la cible ou indisponible au moment de la prévision provoquerait une fuite de données.

Les nombreux candidats IQR ne sont pas supprimés : les différences structurelles entre régions rendent une règle globale insuffisante. Une transformation logarithmique et une analyse par région seront évaluées pendant la préparation ML.

### 9 août 2026 — Sélection et préparation des variables ML

- Ajout du module `src/features/prepare_features.py` pour les quatre tâches : demande de bois de chauffe, demande de charbon, offre de bois de feu et offre de charbon.
- Sélection explicite des variables disponibles avant la prévision et exclusion des identifiants, métadonnées, totaux et autres cibles.
- Exclusion initiale des consommations par habitant pour la demande, car elles participent directement à son calcul.
- Exclusion de `production_charbon_tonnes` et `volume_transporte_tonnes` pour l'offre afin d'éviter une conversion directe de la cible ou une information disponible après la production.
- Ajout de `target_lag_1`, valeur de la cible de l'année précédente, et `target_rolling_mean_3`, moyenne des trois années précédentes. Ces variables n'utilisent jamais la cible de l'année courante.
- Découpage chronologique : entraînement 2000–2018, validation 2019–2021 et test 2022–2024.
- Retrait uniquement des lignes dont la cible de la tâche est absente. Les variables explicatives manquantes sont conservées jusqu'au pipeline scikit-learn.
- Ajout d'un préprocesseur : imputation médiane et standardisation des variables numériques, imputation par modalité fréquente et encodage one-hot des catégories.
- Le préprocesseur doit être ajusté uniquement sur `X_train`, puis appliqué sans réajustement à la validation et au test.

Tailles produites dans `data/processed/` :

- Demande de bois de chauffe : 415 entraînement, 66 validation, 63 test.
- Demande de charbon : 410 entraînement, 66 validation, 64 test.
- Offre de bois de feu : 410 entraînement, 66 validation, 66 test.
- Offre de charbon : 414 entraînement, 65 validation, 66 test.

Le préprocesseur a été vérifié sur les quatre tâches réelles : chaque jeu est transformé en 40 variables numériques et ne contient plus de valeur manquante après transformation. Pour l'évaluation, les retards utilisent uniquement les observations antérieures disponibles. Pour prévoir 2025–2030, les retards devront être alimentés récursivement par les prédictions des années précédentes.

### 9 août 2026 — Premiers modèles d'offre et de demande

- Ajout du module commun `src/models/train_models.py`.
- Comparaison d'une régression Ridge, d'une Random Forest et d'un Gradient Boosting pour chacune des quatre cibles.
- Apprentissage sur `log(1 + cible)` afin de limiter l'effet des fortes asymétries, puis restitution des prédictions en m³ EBR.
- Sélection exclusivement sur le WAPE de validation 2019–2021, avec la MAE comme critère secondaire.
- Réajustement du modèle sélectionné sur 2000–2021, puis évaluation finale unique sur 2022–2024.
- Calcul des métriques MAE, RMSE, R² et WAPE, sauvegarde des pipelines complets, prédictions de test, importances des variables et graphiques observé–prédit.
- Ajout de `requirements.txt` et de tests pour les métriques et la sélection des modèles.

Résultats sur le test :

- Demande de bois de chauffe : Gradient Boosting, WAPE 11,31 %, MAE 67 528 m³ EBR, R² 0,401.
- Demande de charbon : Random Forest, WAPE 11,73 %, MAE 47 149 m³ EBR, R² 0,966.
- Offre de bois de feu : Random Forest, WAPE 12,51 %, MAE 34 121 m³ EBR, R² 0,979.
- Offre de charbon : Random Forest, WAPE 4,84 %, MAE 10 851 m³ EBR, R² 0,998.

La demande de bois de chauffe présente une rupture majeure à Betsiboka : environ 232 706 m³ EBR en 2022, 1 974 787 en 2023, puis 243 746 en 2024. Cette observation explique une grande partie de la RMSE et du faible R² du test. Elle est conservée pour ne pas modifier le test après observation des résultats, mais sa source doit être vérifiée avant toute validation métier.

Les variables temporelles dominent trois modèles. Pour l'offre de bois de feu, la superficie forestière hors aire protégée est la variable principale. Ces performances portent sur des données synthétiques et sur une évaluation à un pas utilisant les observations antérieures disponibles ; elles ne constituent pas encore une validation sur des données réelles.

### 10 août 2026 — Validation temporelle à fenêtres croissantes

- Ajout du module `src/evaluation/time_validation.py`.
- Validation strictement limitée à 2000–2021 ; les années de test 2022–2024 ne participent ni au classement ni aux recommandations.
- Utilisation de trois fenêtres non chevauchantes : 2000–2012 vers 2013–2015, 2000–2015 vers 2016–2018 et 2000–2018 vers 2019–2021.
- Exécution de 36 entraînements : quatre cibles, trois modèles et trois fenêtres.
- Classement par WAPE moyen, puis écart-type du WAPE et MAE moyenne.
- Production des métriques détaillées, résumés de stabilité, prédictions de validation, erreurs régionales et graphique comparatif.

Recommandations issues des trois fenêtres :

- Demande de bois de chauffe : Gradient Boosting, WAPE moyen 6,26 %, écart-type 0,83.
- Demande de charbon : Gradient Boosting provisoire, WAPE moyen 17,45 %, écart-type 7,16.
- Offre de bois de feu : Random Forest, WAPE moyen 11,88 %, écart-type 4,70.
- Offre de charbon : Random Forest, WAPE moyen 4,72 %, écart-type 0,62.

La validation confirme les choix initiaux pour trois cibles. Pour la demande de charbon, Random Forest était meilleure sur 2019–2021, mais Gradient Boosting obtient le meilleur WAPE moyen sur les trois fenêtres. L'écart avec Ridge reste faible et la variabilité est élevée : ce choix est donc provisoire et doit être présenté avec prudence. Analamanga est la région la plus difficile pour la demande de charbon sur les fenêtres évaluées.

Cette validation révèle également que les modèles non linéaires sont nettement supérieurs à Ridge pour les deux offres. Random Forest est particulièrement stable pour l'offre de charbon. Les résultats demeurent fondés sur des données synthétiques.

## Prochaines étapes

- Examiner les rapports qualité et décider du traitement métier des valeurs manquantes.
- Vérifier avec les sources métier les ruptures et valeurs extrêmes, notamment Betsiboka en 2023.
- Figer les modèles recommandés sur toutes les observations historiques 2000–2024, en conservant la réserve sur la demande de charbon.
- Produire les prévisions récursives 2025–2030 pour chaque région et chaque cible.
- Reconstituer les totaux, comparer offre et demande et calculer les écarts régionaux.

## Commandes utiles

```powershell
python -m src.data.clean_data --dataset tous
python -m src.analysis.explore_data --dataset tous
python -m src.features.prepare_features --dataset tous
python -m src.models.train_models --dataset tous
python -m src.evaluation.time_validation --dataset tous
python -m unittest discover -s tests -v
```

Ce fichier doit être mis à jour à chaque fonctionnalité implémentée, changement de données ou décision méthodologique.
