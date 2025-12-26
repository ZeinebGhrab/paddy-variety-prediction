"""
Utilitaires pour sauvegarder et charger les modèles
"""
import pickle
from pathlib import Path
import pandas as pd
import numpy as np

def save_model(model, filename, model_type='regression'):
    """
    Sauvegarde un modèle
    
    Args:
        model: Le modèle à sauvegarder
        filename: Nom du fichier (sans extension)
        model_type: 'regression' ou 'classification'
    """
    model_dir = Path(f"models/{model_type}")
    model_dir.mkdir(parents=True, exist_ok=True)
    
    filepath = model_dir / f"{filename}.pkl"
    
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"✓ Modèle sauvegardé : {filepath}")

def save_scaler(scaler, model_type='regression'):
    """
    Sauvegarde un scaler
    
    Args:
        scaler: Le scaler à sauvegarder
        model_type: 'regression' ou 'classification'
    """
    scaler_dir = Path("models/scalers")
    scaler_dir.mkdir(parents=True, exist_ok=True)
    
    filepath = scaler_dir / f"scaler_{model_type}.pkl"
    
    with open(filepath, 'wb') as f:
        pickle.dump(scaler, f)
    
    print(f"✓ Scaler sauvegardé : {filepath}")

def load_model(filename, model_type='regression'):
    """
    Charge un modèle
    
    Args:
        filename: Nom du fichier (sans extension)
        model_type: 'regression' ou 'classification'
    
    Returns:
        Le modèle chargé
    """
    filepath = Path(f"models/{model_type}/{filename}.pkl")
    
    if not filepath.exists():
        raise FileNotFoundError(f"Modèle non trouvé : {filepath}")
    
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    
    print(f"✓ Modèle chargé : {filepath}")
    return model

def load_scaler(model_type='regression'):
    """
    Charge un scaler
    
    Args:
        model_type: 'regression' ou 'classification'
    
    Returns:
        Le scaler chargé
    """
    filepath = Path(f"models/scalers/scaler_{model_type}.pkl")
    
    if not filepath.exists():
        raise FileNotFoundError(f"Scaler non trouvé : {filepath}")
    
    with open(filepath, 'rb') as f:
        scaler = pickle.load(f)
    
    print(f"✓ Scaler chargé : {filepath}")
    return scaler

def create_model_info_file(models_info, model_type='regression'):
    """
    Crée un fichier JSON avec les informations des modèles
    
    Args:
        models_info: Dictionnaire avec les infos des modèles
        model_type: 'regression' ou 'classification'
    """
    import json
    
    info_dir = Path(f"models/{model_type}")
    info_dir.mkdir(parents=True, exist_ok=True)
    
    filepath = info_dir / "models_info.json"
    
    with open(filepath, 'w') as f:
        json.dump(models_info, f, indent=4)
    
    print(f"✓ Informations sauvegardées : {filepath}")

def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    """
    Évalue un modèle de régression
    
    Args:
        name: Nom du modèle
        model: Le modèle
        X_train, X_test: Features d'entraînement et de test
        y_train, y_test: Labels d'entraînement et de test
    
    Returns:
        Dictionnaire avec les métriques
    """
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from sklearn.model_selection import cross_val_score
    
    # Entraînement
    model.fit(X_train, y_train)
    
    # Prédictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Métriques
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    # Validation croisée
    cv_scores = cross_val_score(model, X_train, y_train, cv=5,
                                scoring='neg_root_mean_squared_error')
    cv_rmse = -cv_scores.mean()
    
    return {
        'Model': name,
        'Train_RMSE': train_rmse,
        'Test_RMSE': test_rmse,
        'Train_MAE': train_mae,
        'Test_MAE': test_mae,
        'Train_R2': train_r2,
        'Test_R2': test_r2,
        'CV_RMSE': cv_rmse
    }

def evaluate_classifier(name, model, X_train, X_test, y_train, y_test):
    """
    Évalue un modèle de classification
    
    Args:
        name: Nom du modèle
        model: Le modèle
        X_train, X_test: Features d'entraînement et de test
        y_train, y_test: Labels d'entraînement et de test
    
    Returns:
        Dictionnaire avec les métriques
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, roc_auc_score
    )
    
    # Entraînement
    model.fit(X_train, y_train)
    
    # Prédictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # ROC-AUC
    roc_auc = None
    if hasattr(model, 'predict_proba'):
        y_pred_proba = model.predict_proba(X_test)
        n_classes = len(np.unique(y_test))
        
        if n_classes == 2:
            roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
        else:
            roc_auc = roc_auc_score(
                y_test,
                y_pred_proba,
                multi_class='ovr',
                average='weighted'
            )
    
    return {
        'Model': name,
        'Accuracy_Train': accuracy_score(y_train, y_pred_train),
        'Accuracy_Test': accuracy_score(y_test, y_pred_test),
        'Precision': precision_score(y_test, y_pred_test, average='weighted'),
        'Recall': recall_score(y_test, y_pred_test, average='weighted'),
        'F1_Score': f1_score(y_test, y_pred_test, average='weighted'),
        'ROC_AUC': roc_auc
    }

if __name__ == "__main__":
    print("Module d'utilitaires pour les modèles")
    print("Importez les fonctions nécessaires dans vos scripts")