# 🌾 Paddy Rice Yield Prediction & Farmer Profiling

Projet d'analyse de données agricoles combinant modélisation prédictive et segmentation pour optimiser la production rizicole en Inde.

---

## 📋 Table des Matières

- [Vue d'ensemble](#-vue-densemble)  
- [Objectifs du Projet](#-objectifs-du-projet)  
- [Données](#-données)  
- [Méthodologie](#-méthodologie)  
- [Résultats Clés](#-résultats-clés)  
- [Structure du Projet](#-structure-du-projet)  
- [Installation](#-installation)  
- [Contribution](#-contribution)  
- [Licence](#-licence)  
- [À propos du développeur](#-a-propos-du-developpeur)  

---

## 🎯 Vue d'ensemble

Ce projet analyse 2 790 parcelles de riz en Inde pour :

- Prédire le rendement (kg/parcelle) à partir des pratiques agricoles  
- Identifier 6 profils d'agriculteurs distincts via clustering  

Le rendement varie de 5 000 à 40 000 kg/parcelle, reflétant des stratégies de gestion très contrastées. L'objectif est de transformer ces données en leviers d'optimisation concrète pour améliorer la productivité agricole.

---

## 🎯 Objectifs du Projet

### 1️⃣ Modélisation Prédictive (Régression)

- Construire un modèle pour prédire le rendement en riz (variable cible : `Paddy yield(in Kg)`)  
- Quantifier l'impact de chaque décision agricole (engrais, pesticides, irrigation)  
- Identifier les variables les plus influentes sur la production  

### 2️⃣ Segmentation des Agriculteurs (Clustering)

- Découvrir des profils agricoles homogènes (intensif, optimal, économe...)  
- Extraire des recommandations personnalisées par profil  
- Détecter les anomalies (parcelles inefficaces malgré des intrants élevés)  

---

## 📊 Données

**Source :** Dataset Paddy Rice (Inde) contenant 2 790 observations et 45 variables.  

| Catégorie       | Exemples de Variables                      |
|-----------------|-------------------------------------------|
| Intrants        | DAP, Urée, Potasse, Pesticides, Micronutriments |
| Pratiques       | Paille recyclée, Densité de semis, Surface cultivée |
| Environnement   | Température (min/max), Pluviométrie (30j, 70j), Type de sol |
| Rendement       | Paddy yield(in Kg) ⭐ (variable cible)     |

**Fichiers de Données :**

```
data/
├── paddydataset.csv
├── noisy_paddydataset.csv
├── cleaned_paddydataset.csv
└── paddy_dataset_fe.csv
```


---

## 🔬 Méthodologie

### Phase 1 : Exploration & Nettoyage
- EDA approfondie : distributions, corrélations, outliers  
- Nettoyage : gestion des valeurs manquantes, détection d'anomalies  
- Feature Engineering : création de variables dérivées (ratios, agrégations temporelles)  

### Phase 2 : Modélisation Régression
**Feature Selection :**
- SelectKBest : sélection des 12 meilleures features  
- Backward Elimination : sélection basée sur p-values (OLS)  

**Modèles Testés :**
- Régression Linéaire  
- Lasso (L1)  
- Ridge (L2)  
- ElasticNet (L1 + L2)  
- XGBoost Regressor ⭐  

**Évaluation :**
- RMSE, MAE, R²  
- Validation croisée (5-fold)  
- Analyse résiduelle  

### Phase 3 : Clustering
**Réduction Dimensionnelle :**
- PCA : 47,5 % de variance expliquée avec 2 composantes  
- UMAP : préservation de la structure non-linéaire  
- t-SNE : visualisation des clusters  

**Algorithmes :**
- K-Means (principal)  
- Clustering Hiérarchique  
- GMM (Gaussian Mixture Model)  

**Optimisation du Nombre de Clusters :**
- Elbow Method + KneeLocator  
- Silhouette Score  
- BIC/AIC pour GMM  
- Résultat optimal : k=6  

**Interprétation :**
- Analyse des centroïdes  
- Heatmap des profils  
- Decision Tree pour extraction de règles  

---

## 🏆 Résultats Clés

### 📈 Régression : Prédiction du Rendement

| Modèle          | RMSE Test | MAE Test | R² Test | Verdict |
|-----------------|-----------|----------|---------|---------|
| Linear Regression | 3130.25  | 1652.84  | 0.8754  | Bon     |
| Lasso           | 3130.86  | 1652.96  | 0.8753  | Bon     |
| Ridge           | 3130.13  | 1652.65  | 0.8754  | Bon     |
| ElasticNet      | 3130.48  | 1652.72  | 0.8754  | Bon     |
| XGBoost         | 2938.82  | 1550.29  | 0.8830  | 🥇 Meilleur |

**Variables les Plus Importantes :**
- Température maximale (J61-J90)  
- Pluviométrie (70 jours)  
- DAP (engrais phosphaté)  
- Urée (fertilisant azoté)  
- Paille recyclée  

### 🔍 Clustering : 6 Profils Agricoles Identifiés

| Cluster | Profil         | Effectif | Rendement Moyen | Caractéristiques                    |
|---------|----------------|----------|----------------|-------------------------------------|
| 0       | 🏆 Champion Intensif | 605      | 23 121 kg     | Intrants élevés, paille maximale    |
| 5       | 🥈 Champion (variante) | 486      | 23 025 kg     | Similaire au Cluster 0              |
| 3       | ⭐ Profil Optimal | 425      | 22 619 kg     | Bon rendement, moins de ressources  |
| 1       | 📊 Standard     | 477      | 22 525 kg     | Pratiques moyennes                  |
| 4       | 📊 Standard     | 386      | 22 525 kg     | Pratiques moyennes                  |
| 2       | ⚠️ Économe      | 410      | 22 409 kg     | Sous-investissement en intrants     |

**Score de Silhouette :**
- K-Means : 0.3449  
- Clustering Hiérarchique : 0.3337  
- GMM : 0.3449  

---

## 📁 Structure du Projet

```
paddy-variety-prediction/
│
├── data/                           # Données
│   ├── paddydataset.csv            # Dataset original
│   ├── noisy_paddydataset.csv      # Dataset avec bruit
│   ├── paddy_dataset_fe.csv        # # Dataset après ingénierie des caractéristiques
│   └── cleaned_paddydataset.csv    # Dataset nettoyé
│
│
├── src/                            # Scripts de traitement
│   ├── 01_eda.py                   # Analyse exploratoire
│   ├── 02_data_cleaning.py         # Nettoyage des données
│   ├── 02_feature_engineering.py   # Engineering des features
│   ├── 04_regression_modeling.py   # Modélisation régression
│   └── 05_clustering_modeling.py   # Modélisation clustering
│                    
├── results/ 
│   ├── cluster_assignments.scv     # Attribution des parcelles aux clusters   
│   ├── cluster_centroids.csv       # Profils moyens de chaque cluster 
│   └── cluster_statistics.csv      # Statistiques détaillées par cluster
│
├── requirements.txt                # Dépendances
└── README.md                     

```


---

## 🛠️ Installation

**Prérequis :**
- Python 3.8+  
- pip  

**Étapes :**
```bash
# 1. Cloner le repository
git clone https://github.com/ZeinebGhrab/paddy-variety-prediction.git
cd paddy-variety-prediction
```

**2. Créer un environnement virtuel (recommandé)**
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```
**3. Installer les dépendances**
```bash
pip install -r requirements.txt
```
---

## 🤝 Contribution

Les contributions sont les bienvenues ! Pour contribuer :

1. Forkez le projet
2. Créez une branche (`git checkout -b feature/AmazingFeature`)
3. Committez vos changements (`git commit -m 'Add AmazingFeature'`)
4. Pushez vers la branche (`git push origin feature/AmazingFeature`)
5. Ouvrez une Pull Request

## 📄 Licence

MIT License © Zeineb Ghrab

## 🙋 À propos du développeur  
Réalisée avec passion par Zeineb Ghrab  
🎓 Ingénieure en Data Science | 🧠 Passionnée par les données, l'IA et le développement full-stack
