# Suivi du projet bois-énergie

Dernière mise à jour : 4 août 2026

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

## Prochaines étapes

- Examiner les rapports qualité et décider du traitement métier des valeurs manquantes.
- Sélectionner les variables disponibles au moment de la prévision et éliminer les risques de fuite de données.
- Construire les variables temporelles sans fuite de données.
- Préparer les jeux chronologiques d'entraînement, validation et test avec imputation ajustée sur l'entraînement.
- Entraîner les modèles de référence puis comparer les modèles ML classiques.

## Commandes utiles

```powershell
python -m src.data.clean_data --dataset tous
python -m src.analysis.explore_data --dataset tous
python -m unittest discover -s tests -v
```

Ce fichier doit être mis à jour à chaque fonctionnalité implémentée, changement de données ou décision méthodologique.
