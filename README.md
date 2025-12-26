# 🌾 Système d'Aide à la Décision Agricole - Culture du Riz

## 📋 Description

Application complète d'aide à la décision pour les agriculteurs cultivant le riz. Le système utilise des modèles de Machine Learning pour :
- **Prédire le rendement** des cultures
- **Recommander la variété** de riz optimale
- **Analyser les données** agronomiques

## 🚀 Installation

### Prérequis
- Python 3.8 ou supérieur
- pip

### Étapes d'installation

1. **Cloner le repository**
```bash
git clone https://github.com/ZeinebGhrab/paddy-variety-prediction.git
cd paddy_project
```

2. **Créer un environnement virtuel**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows
```

3. **Installer les dépendances**
```bash
pip install -r requirements.txt
```

## 📁 Structure du Projet

```
paddy-variety-prediction/
│
├── data/                          # Données
│   ├── paddydataset.csv          # Dataset original
│   ├── noisy_paddydataset.csv    # Dataset avec bruit
│   └── cleaned_paddydataset.csv  # Dataset nettoyé
│
├── models/                        # Modèles sauvegardés
│   ├── regression/               # Modèles de régression
│   ├── classification/           # Modèles de classification
│   └── scalers/                  # Scalers
│
├── src/                          # Scripts de traitement
│   ├── 01_data_generation.py    # Génération données bruitées
│   ├── 02_eda.py                # Analyse exploratoire
│   ├── 03_data_cleaning.py      # Nettoyage des données
│   ├── 04_feature_engineering.py # Engineering des features
│   ├── 05_regression_modeling.py # Modélisation régression
│   ├── 06_classification_modeling.py # Modélisation classification
│   └── utils.py                  # Utilitaires
│
├── app/                          # Application Streamlit
│   ├── streamlit_app.py         # Page principale
│   └── pages/                    # Pages de l'application
│       ├── 1_🌾_Prédiction_Rendement.py
│       ├── 2_🌱_Recommandation_Variété.py
│       └── 3_📊_Analyse_Données.py
│
├── requirements.txt              # Dépendances
└── README.md                     # Ce fichier
```

## 🎯 Utilisation

### 1. Préparation des données

```bash
# Générer les données bruitées
python src/01_data_generation.py

# Analyse exploratoire
python src/02_eda.py

# Nettoyage
python src/03_data_cleaning.py
```

### 2. Entraînement des modèles

```bash
# Feature engineering
python src/04_feature_engineering.py

# Modèles de régression
python src/05_regression_modeling.py

# Modèles de classification
python src/06_classification_modeling.py
```

### 3. Lancement de l'application

```bash
streamlit run app/streamlit_app.py
```

L'application sera accessible à l'adresse : `http://localhost:8501`

## 📊 Modèles Disponibles

### Régression (Prédiction de Rendement)
- **Ridge Regression** (Recommandé)
  - R² = 0.89
  - RMSE = 2887 kg
  - MAE = 1688 kg

- **XGBoost Regressor**
  - R² = 0.90
  - RMSE = 2665 kg
  - MAE = 1550 kg

- Linear Regression, Lasso, ElasticNet

### Classification (Recommandation de Variété)
- **XGBoost Classifier** (Recommandé)
  - Accuracy = 87%
  - F1-Score = 0.87
  - ROC-AUC = 0.87

- Random Forest, Logistic Regression, KNN, Decision Tree

## 🌱 Variétés de Riz

### CO_43
- Résistant à la sécheresse
- Cycle : 130-135 jours
- Rendement : 3500-4000 kg/ha
- Sol idéal : Alluvial

### Ponmani
- Qualité premium
- Cycle : 145-150 jours
- Rendement : 4000-4500 kg/ha
- Sol idéal : Argileux

### Delux Ponni
- Haut rendement
- Cycle : 135-140 jours
- Rendement : 4200-4800 kg/ha
- Sol idéal : Polyvalent

## 📈 Fonctionnalités de l'Application

### Page 1 : Prédiction du Rendement
- Saisie des données de la parcelle
- Prédiction du rendement en kg
- Recommandations personnalisées
- Comparaison avec les moyennes

### Page 2 : Recommandation de Variété
- Analyse des conditions de culture
- Recommandation de la variété optimale
- Niveau de confiance
- Comparaison des 3 variétés

### Page 3 : Analyse des Données
- Visualisations interactives
- Statistiques descriptives
- Corrélations
- Export des résultats

## 🔧 Configuration Avancée

### Réentraîner les modèles

Si vous souhaitez réentraîner les modèles avec de nouvelles données :

```python
from src.utils import save_model, save_scaler
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

# Entraîner votre modèle
model = XGBRegressor()
model.fit(X_train, y_train)

# Sauvegarder
save_model(model, 'xgboost_reg', model_type='regression')

# Sauvegarder le scaler
scaler = StandardScaler()
scaler.fit(X_train)
save_scaler(scaler, model_type='regression')
```

### Ajouter un nouveau modèle

1. Entraîner et sauvegarder le modèle
2. Ajouter les performances dans `performance_metrics`
3. Ajouter dans le selectbox de l'interface

## 📝 Variables d'Entrée

### Données Météorologiques
- Précipitations par période (mm)
- Températures min/max (°C)
- Humidité (%)
- Vitesse du vent (km/h)

### Caractéristiques de la Parcelle
- Superficie (hectares)
- Type de sol
- Surface de pépinière
- Bloc agricole

### Intrants
- DAP, Urée, Potasse (kg)
- Micronutriments (kg)
- Pesticides, herbicides (ml)
- Taux de semis (kg)

## 🤝 Contribution

Les contributions sont les bienvenues ! Pour contribuer :

1. Forkez le projet
2. Créez une branche (`git checkout -b feature/AmazingFeature`)
3. Committez vos changements (`git commit -m 'Add AmazingFeature'`)
4. Pushez vers la branche (`git push origin feature/AmazingFeature`)
5. Ouvrez une Pull Request

## 📧 Contact

Pour toute question ou support :
- 📧 Email : support@paddy-ai.tn
- 🌐 Web : www.paddy-ai.tn

## 📄 Licence

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.

## 🙏 Remerciements

- Agriculteurs participants pour les données
- Ministère de l'Agriculture pour le soutien
- Communauté open source pour les outils

---

**Développé avec ❤️ pour les agriculteurs tunisiens**