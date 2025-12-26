import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats.mstats import winsorize
import matplotlib.pyplot as plt
import seaborn as sns

def impute_numerical_features(df, numeric_features):
    """Imputation des valeurs manquantes pour les variables numériques"""
    print("IMPUTATION DES VARIABLES NUMÉRIQUES")
    print("="*80)
    
    imputation_suggestions = []
    
    for col in numeric_features:
        data = df[col].dropna()
        n = len(data)
        
        if n < 3:
            continue
        
        mean = data.mean()
        median = data.median()
        skewness = stats.skew(data)
        kurt = stats.kurtosis(data)
        
        # Test de normalité
        if n <= 500:
            shapiro_stat, shapiro_p = stats.shapiro(data)
            test_name = 'Shapiro-Wilk'
            p_value = shapiro_p
        elif n <= 5000:
            standardized_data = (data - mean) / data.std()
            ks_stat, ks_p = stats.kstest(standardized_data, 'norm')
            test_name = 'Kolmogorov-Smirnov'
            p_value = ks_p
        else:
            p_value = np.nan
            test_name = 'N/A'
        
        # Décision d'imputation
        if abs(skewness) <= 0.5 and (np.isnan(p_value) or p_value > 0.05):
            imputation = "Moyenne (distribution symétrique et normale)"
            df[col].fillna(df[col].mean(), inplace=True)
        else:
            imputation = "Médiane (distribution asymétrique ou non normale)"
            df[col].fillna(df[col].median(), inplace=True)
        
        imputation_suggestions.append({
            'Colonne': col,
            'Moyenne': round(mean, 2),
            'Médiane': round(median, 2),
            'Skewness': round(skewness, 3),
            'Test Normalité': test_name,
            'p-value': round(p_value, 4) if not np.isnan(p_value) else 'N/A',
            'Suggestion Imputation': imputation
        })
        
        print(f"{col} : imputé par {'MOYENNE' if 'Moyenne' in imputation else 'MÉDIANE'}")
    
    print(f"\nValeurs manquantes restantes (numériques) : {df[numeric_features].isnull().sum().sum()}")
    
    results_df = pd.DataFrame(imputation_suggestions)
    return df, results_df

def treat_outliers(df, numeric_features, min_rows_threshold=5000, plot=False):
    """Traite les outliers automatiquement"""
    print("\nTRAITEMENT AUTOMATIQUE DES OUTLIERS")
    print("="*100)
    
    summary = []
    rows_to_drop = set()
    initial_rows = len(df)
    df_before = df.copy()
    
    for col in numeric_features:
        data = df[col].dropna()
        
        if len(data) < 10:
            summary.append({
                'Colonne': col,
                'Méthode Choisie': 'Ignoré (trop peu de données)'
            })
            continue
        
        skewness = stats.skew(data)
        median = data.median()
        
        # Détection IQR
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        
        outlier_mask = (df[col] < lower) | (df[col] > upper)
        nb_outliers = outlier_mask.sum()
        percent_outliers = nb_outliers / len(df) * 100
        
        method_chosen = ""
        action = ""
        
        if nb_outliers == 0:
            method_chosen = "Aucun outlier"
            action = "Rien à faire"
        
        elif percent_outliers < 1 and len(df) > min_rows_threshold:
            rows_to_drop.update(df.loc[outlier_mask].index)
            method_chosen = "Suppression"
            action = f"{nb_outliers} lignes supprimées"
        
        elif abs(skewness) > 1 and data.min() >= 0:
            df[col + "_log"] = np.log1p(df[col])
            method_chosen = "Transformation log"
            action = "Nouvelle colonne _log créée"
        
        elif percent_outliers > 5:
            df[col] = winsorize(df[col], limits=[0.05, 0.05])
            method_chosen = "Capping (5%/95%)"
            action = "Winsorization appliquée"
        
        else:
            df.loc[outlier_mask, col] = median
            method_chosen = "Robust IQR + médiane"
            action = f"{nb_outliers} outliers remplacés par médiane"
        
        summary.append({
            'Colonne': col,
            'Skewness': round(skewness, 3),
            '% Outliers': round(percent_outliers, 2),
            'Méthode Choisie': method_chosen,
            'Action': action
        })
    
    # Suppression finale
    if rows_to_drop:
        df.drop(index=list(rows_to_drop), inplace=True)
        df.reset_index(drop=True, inplace=True)
    
    final_rows = len(df)
    print(f"\nNombre de lignes : {initial_rows} → {final_rows}")
    
    summary_df = pd.DataFrame(summary)
    print("\nRÉCAPITULATIF DU TRAITEMENT AUTOMATIQUE")
    print(summary_df.to_string(index=False))
    
    return df, summary_df

def clean_categorical_features(df, categorical_features):
    """Nettoie et impute les variables catégorielles"""
    print("\nTRAITEMENT DES VARIABLES CATÉGORIELLES")
    print("="*80)
    
    treatment_summary = []
    
    for col in categorical_features:
        if col not in df.columns:
            print(f"⚠️ {col} : colonne non présente")
            continue
        
        initial_missing = df[col].isnull().sum()
        initial_unique = df[col].nunique()
        
        # Imputation par le mode
        if initial_missing > 0:
            mode_value = df[col].mode(dropna=True)[0]
            df[col].fillna(mode_value, inplace=True)
            imputed = True
        else:
            mode_value = None
            imputed = False
        
        # Uniformisation
        df[col] = df[col].astype(str).str.lower().str.strip()
        
        final_missing = df[col].isnull().sum()
        final_unique = df[col].nunique()
        
        print(f"✓ {col}")
        if imputed:
            print(f"   → {initial_missing} valeurs manquantes imputées par mode : '{mode_value}'")
        print(f"   → Valeurs uniques : {initial_unique} → {final_unique}")
        
        treatment_summary.append({
            'Colonne': col,
            'Manquantes Initiales': initial_missing,
            'Imputé par Mode': mode_value if imputed else 'Non',
            'Uniques Avant': initial_unique,
            'Uniques Après': final_unique
        })
    
    total_missing_cat = df[categorical_features].isnull().sum().sum()
    print(f"\nValeurs manquantes restantes (catégorielles) : {total_missing_cat}")
    
    summary_df = pd.DataFrame(treatment_summary)
    return df, summary_df

if __name__ == "__main__":
    # Charger les données
    df = pd.read_csv('data/noisy_paddydataset.csv')
    
    # Identifier les types de variables
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = df.select_dtypes(include=['object']).columns.tolist()
    
    # Imputation numériques
    df, imputation_results = impute_numerical_features(df, numeric_features)
    
    # Traitement des outliers
    df, outlier_results = treat_outliers(df, numeric_features)
    
    # Nettoyage catégorielles
    df, categorical_results = clean_categorical_features(df, categorical_features)
    
    # Sauvegarder
    df.to_csv('data/cleaned_paddydataset.csv', index=False, encoding='utf-8')
    
    print("\n" + "="*80)
    print("Dataset nettoyé enregistré avec succès !")
    print(f"Fichier : data/cleaned_paddydataset.csv")
    print(f"Dimensions finales : {df.shape}")
    print("="*80)