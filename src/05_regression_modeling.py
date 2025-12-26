"""
Script pour entraîner et sauvegarder tous les modèles de régression
Ce script reprend exactement votre code original
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
from utils import save_model, save_scaler, evaluate_model, create_model_info_file

print("="*80)
print("ENTRAÎNEMENT DES MODÈLES DE RÉGRESSION")
print("="*80)

# 1. Charger les données nettoyées
df = pd.read_csv('data/cleaned_paddydataset.csv')
print(f"\nDonnées chargées : {df.shape}")

# 2. Préparation des données
target = 'Paddy yield(in Kg)'
y = df[target]
X = df.drop(target, axis=1)

# Encodage one-hot des variables catégorielles
categorical_features = X.select_dtypes(include=['object']).columns.tolist()
X = pd.get_dummies(X, columns=categorical_features, drop_first=False, dtype=int)

print(f"Features après encodage : {X.shape}")

# 3. Feature Selection avec SelectKBest
print("\n" + "-"*80)
print("SÉLECTION DES FEATURES")
print("-"*80)

selector = SelectKBest(score_func=f_regression, k=12)
X_selected = selector.fit_transform(X, y)
selected_features = X.columns[selector.get_support()].tolist()

print(f"Features sélectionnées ({len(selected_features)}) :")
for feat in selected_features:
    print(f"  - {feat}")

# Créer DataFrame avec features sélectionnées
df_selected = pd.DataFrame(X_selected, columns=selected_features)
df_selected[target] = y.values

# 4. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    df_selected.drop(target, axis=1),
    df_selected[target],
    test_size=0.2,
    random_state=42
)

print(f"\nTrain set : {X_train.shape}")
print(f"Test set  : {X_test.shape}")

# 5. Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n✓ Standardisation effectuée")

# 6. Entraînement des modèles
print("\n" + "="*80)
print("ENTRAÎNEMENT DES MODÈLES")
print("="*80)

results = {}
models_to_save = {}

# Linear Regression
print("\n1. LINEAR REGRESSION")
print("-" * 35)
lr_model = LinearRegression()
results['Linear Regression'] = evaluate_model(
    'Linear Regression', lr_model,
    X_train_scaled, X_test_scaled, y_train, y_test
)
models_to_save['linear_regression'] = lr_model
print(f"  Test RMSE : {results['Linear Regression']['Test_RMSE']:.4f}")
print(f"  Test R²   : {results['Linear Regression']['Test_R2']:.4f}")

# Lasso
print("\n2. LASSO REGRESSION")
print("-" * 35)
lasso_model = Lasso(alpha=0.001, max_iter=10000, random_state=42)
results['Lasso'] = evaluate_model(
    'Lasso', lasso_model,
    X_train_scaled, X_test_scaled, y_train, y_test
)
models_to_save['lasso'] = lasso_model
print(f"  Test RMSE : {results['Lasso']['Test_RMSE']:.4f}")
print(f"  Test R²   : {results['Lasso']['Test_R2']:.4f}")

# Ridge
print("\n3. RIDGE REGRESSION")
print("-" * 35)
ridge_model = Ridge(alpha=10, random_state=42)
results['Ridge'] = evaluate_model(
    'Ridge', ridge_model,
    X_train_scaled, X_test_scaled, y_train, y_test
)
models_to_save['ridge'] = ridge_model
print(f"  Test RMSE : {results['Ridge']['Test_RMSE']:.4f}")
print(f"  Test R²   : {results['Ridge']['Test_R2']:.4f}")

# ElasticNet
print("\n4. ELASTIC NET")
print("-" * 35)
elasticnet_model = ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=10000, random_state=42)
results['ElasticNet'] = evaluate_model(
    'ElasticNet', elasticnet_model,
    X_train_scaled, X_test_scaled, y_train, y_test
)
models_to_save['elasticnet'] = elasticnet_model
print(f"  Test RMSE : {results['ElasticNet']['Test_RMSE']:.4f}")
print(f"  Test R²   : {results['ElasticNet']['Test_R2']:.4f}")

# XGBoost
print("\n5. XGBOOST REGRESSOR")
print("-" * 35)
xgb_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=500,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
results['XGBoost'] = evaluate_model(
    'XGBoost', xgb_model,
    X_train_scaled, X_test_scaled, y_train, y_test
)
models_to_save['xgboost_reg'] = xgb_model
print(f"  Test RMSE : {results['XGBoost']['Test_RMSE']:.4f}")
print(f"  Test R²   : {results['XGBoost']['Test_R2']:.4f}")

# 7. Comparaison et affichage
print("\n" + "="*80)
print("COMPARAISON DES MODÈLES")
print("="*80)

comparison_df = pd.DataFrame(results).T
print(comparison_df[['Test_RMSE', 'Test_MAE', 'Test_R2', 'CV_RMSE']].to_string())

# 8. Sauvegarde des modèles
print("\n" + "="*80)
print("SAUVEGARDE DES MODÈLES")
print("="*80)

for name, model in models_to_save.items():
    save_model(model, name, model_type='regression')

# Sauvegarder le scaler
save_scaler(scaler, model_type='regression')

# Créer fichier d'informations
models_info = {}
for model_name, metrics in results.items():
    models_info[model_name] = {
        'test_rmse': float(metrics['Test_RMSE']),
        'test_mae': float(metrics['Test_MAE']),
        'test_r2': float(metrics['Test_R2']),
        'cv_rmse': float(metrics['CV_RMSE']),
        'features': selected_features
    }

create_model_info_file(models_info, model_type='regression')

print("\n" + "="*80)
print("✓ TOUS LES MODÈLES ONT ÉTÉ SAUVEGARDÉS AVEC SUCCÈS")
print("="*80)

# Meilleur modèle
best_model = comparison_df['Test_R2'].idxmax()
best_r2 = comparison_df.loc[best_model, 'Test_R2']
best_rmse = comparison_df.loc[best_model, 'Test_RMSE']

print(f"\n🏆 Meilleur modèle : {best_model}")
print(f"   R² Test    : {best_r2:.4f}")
print(f"   RMSE Test  : {best_rmse:.4f} kg")
print(f"   Fichier    : models/regression/{best_model.lower().replace(' ', '_')}.pkl")