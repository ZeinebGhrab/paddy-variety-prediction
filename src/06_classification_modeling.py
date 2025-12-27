"""
Script pour entraîner et sauvegarder tous les modèles de classification
Ce script reprend avec détection automatique de Variety
"""
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier
from utils import save_model, save_scaler, evaluate_classifier, create_model_info_file
import sys

print("="*80)
print("ENTRAÎNEMENT DES MODÈLES DE CLASSIFICATION")
print("="*80)

# 1. Charger les données
df = pd.read_csv('data/cleaned_paddydataset.csv')
print(f"\nDonnées chargées : {df.shape}")
print(f"Colonnes disponibles : {list(df.columns)}")

# 2. Identifier la colonne Variety
variety_col = None
for col in df.columns:
    if 'variety' in col.lower():
        variety_col = col
        break

if variety_col is None:
    print("\nERREUR: Aucune colonne 'Variety' trouvée dans le dataset")
    print("Colonnes disponibles:")
    for col in df.columns:
        print(f"  - {col}")
    exit(1)

print(f"\n✓ Colonne cible trouvée : {variety_col}")

# 3. Vérifier les valeurs de Variety
print(f"\nValeurs uniques de {variety_col} : {df[variety_col].unique()}")
print(f"Distribution:")
print(df[variety_col].value_counts())

# 4. Séparation X et y
y = df[variety_col].copy()
X = df.drop(variety_col, axis=1)

# Encodage des variables catégorielles dans X (sauf Variety)
categorical_features = X.select_dtypes(include=['object']).columns.tolist()
if categorical_features:
    print(f"\nEncodage de {len(categorical_features)} variables catégorielles...")
    X = pd.get_dummies(X, columns=categorical_features, drop_first=False, dtype=int)

print(f"\nShape de X : {X.shape}")
print(f"Shape de y : {y.shape}")

# 5. Encoder la cible (Variety) en numérique si nécessaire
from sklearn.preprocessing import LabelEncoder

if y.dtype == 'object':
    label_encoder = LabelEncoder()
    y_labels = label_encoder.fit_transform(y)
    variety_names = label_encoder.classes_.tolist()
    print(f"\n✓ Encodage de la cible effectué")
    print(f"Mapping : {dict(enumerate(variety_names))}")
else:
    y_labels = y.values
    variety_names = sorted(y.unique())

print(f"\nVariétés : {variety_names}")
print(f"Distribution : {pd.Series(y_labels).value_counts().to_dict()}")

# 5. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_labels,
    test_size=0.2,
    random_state=42,
    stratify=y_labels
)

print(f"\nTrain set : {X_train.shape}")
print(f"Test set  : {X_test.shape}")

# 6. Standardisation
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("✓ Standardisation effectuée")

# 7. Entraînement des modèles
print("\n" + "="*80)
print("ENTRAÎNEMENT DES MODÈLES")
print("="*80)

results = {}
models_to_save = {}

# K-Nearest Neighbors
print("\n1. K-NEAREST NEIGHBORS")
print("-" * 35)
knn = KNeighborsClassifier(n_neighbors=5)
results['KNN'] = evaluate_classifier(
    'K-Nearest Neighbors', knn,
    X_train_scaled, X_test_scaled, y_train, y_test
)
models_to_save['knn'] = knn
print(f"  Accuracy Test : {results['KNN']['Accuracy_Test']:.4f}")
print(f"  F1-Score      : {results['KNN']['F1_Score']:.4f}")

# Logistic Regression
print("\n2. LOGISTIC REGRESSION")
print("-" * 35)
lr = LogisticRegression(max_iter=1000, random_state=42)
results['Logistic Regression'] = evaluate_classifier(
    'Logistic Regression', lr,
    X_train_scaled, X_test_scaled, y_train, y_test
)
models_to_save['logistic_regression'] = lr
print(f"  Accuracy Test : {results['Logistic Regression']['Accuracy_Test']:.4f}")
print(f"  F1-Score      : {results['Logistic Regression']['F1_Score']:.4f}")

# Decision Tree
print("\n3. DECISION TREE")
print("-" * 35)
dt = DecisionTreeClassifier(max_depth=5, random_state=42)
results['Decision Tree'] = evaluate_classifier(
    'Decision Tree', dt,
    X_train_scaled, X_test_scaled, y_train, y_test
)
models_to_save['decision_tree'] = dt
print(f"  Accuracy Test : {results['Decision Tree']['Accuracy_Test']:.4f}")
print(f"  F1-Score      : {results['Decision Tree']['F1_Score']:.4f}")

# Random Forest
print("\n4. RANDOM FOREST")
print("-" * 35)
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
results['Random Forest'] = evaluate_classifier(
    'Random Forest', rf,
    X_train_scaled, X_test_scaled, y_train, y_test
)
models_to_save['random_forest'] = rf
print(f"  Accuracy Test : {results['Random Forest']['Accuracy_Test']:.4f}")
print(f"  F1-Score      : {results['Random Forest']['F1_Score']:.4f}")

# XGBoost
print("\n5. XGBOOST CLASSIFIER")
print("-" * 35)
xgb_clf = XGBClassifier(n_estimators=100, random_state=42, n_jobs=-1, eval_metric='logloss')
results['XGBoost'] = evaluate_classifier(
    'XGBoost', xgb_clf,
    X_train_scaled, X_test_scaled, y_train, y_test
)
models_to_save['xgboost_clf'] = xgb_clf
print(f"  Accuracy Test : {results['XGBoost']['Accuracy_Test']:.4f}")
print(f"  F1-Score      : {results['XGBoost']['F1_Score']:.4f}")

# 8. Comparaison
print("\n" + "="*80)
print("COMPARAISON DES MODÈLES")
print("="*80)

comparison_df = pd.DataFrame(results).T
print(comparison_df[['Accuracy_Test', 'Precision', 'Recall', 'F1_Score', 'ROC_AUC']].to_string())

# 9. Sauvegarde
print("\n" + "="*80)
print("SAUVEGARDE DES MODÈLES")
print("="*80)

for name, model in models_to_save.items():
    save_model(model, name, model_type='classification')

# Sauvegarder le scaler
save_scaler(scaler, model_type='classification')

# Sauvegarder le LabelEncoder
import pickle
from pathlib import Path

encoder_path = Path("models/classification/label_encoder.pkl")
with open(encoder_path, 'wb') as f:
    if y.dtype == 'object':
        pickle.dump(label_encoder, f)
    else:
        pickle.dump(None, f)
print(f"✓ Label encoder sauvegardé : {encoder_path}")

# Créer fichier d'informations
models_info = {}
for model_name, metrics in results.items():
    models_info[model_name] = {
        'accuracy_test': float(metrics['Accuracy_Test']),
        'precision': float(metrics['Precision']),
        'recall': float(metrics['Recall']),
        'f1_score': float(metrics['F1_Score']),
        'roc_auc': float(metrics['ROC_AUC']) if metrics['ROC_AUC'] is not None else None,
        'variety_names': variety_names
    }

create_model_info_file(models_info, model_type='classification')

print("\n" + "="*80)
print("✓ TOUS LES MODÈLES ONT ÉTÉ SAUVEGARDÉS AVEC SUCCÈS")
print("="*80)

# Meilleur modèle
best_model = comparison_df['F1_Score'].idxmax()
best_f1 = comparison_df.loc[best_model, 'F1_Score']
best_acc = comparison_df.loc[best_model, 'Accuracy_Test']

print(f"\n🏆 Meilleur modèle : {best_model}")
print(f"   Accuracy Test : {best_acc:.4f}")
print(f"   F1-Score      : {best_f1:.4f}")
print(f"   Fichier       : models/classification/{best_model.lower().replace(' ', '_')}.pkl")