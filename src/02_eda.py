import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Configuration
plt.rcParams['figure.figsize'] = (12, 8)
sns.set_style("whitegrid")
sns.set_palette("husl")

def load_data(filepath):
    """Charge les données"""
    df = pd.read_csv(filepath)
    print("="*60)
    print("APERÇU GÉNÉRAL DU DATASET")
    print("="*60)
    print(f"Dimensions du dataset: {df.shape}")
    print(f"Nombre total de cellules: {df.shape[0] * df.shape[1]:,}")
    print(df.head())
    return df

def analyze_missing_values(df):
    """Analyse des valeurs manquantes"""
    print("\n" + "="*60)
    print("ANALYSE DES DONNÉES MANQUANTES")
    print("="*60)
    
    missing_data = df.isnull().sum()
    missing_percent = (missing_data / len(df)) * 100
    missing_df = pd.DataFrame({
        'Colonnes': missing_data.index,
        'Valeurs_Manquantes': missing_data.values,
        'Pourcentage': missing_percent.values
    }).sort_values('Pourcentage', ascending=False)
    
    missing_cols = missing_df[missing_df['Valeurs_Manquantes'] > 0]
    print(f"Nombre de colonnes avec des valeurs manquantes: {len(missing_cols)}")
    print("\nColonnes avec le plus de valeurs manquantes:")
    print(missing_cols.head(10))
    
    return missing_df

def identify_feature_types(df):
    """Identifie les types de variables"""
    print("\n" + "-"*70)
    print("DISTINCTION VARIABLES NUMÉRIQUES ET CATÉGORIQUES")
    print("-"*70)
    
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = df.select_dtypes(include=['object']).columns.tolist()
    
    print(f"Variables numériques: {len(numeric_features)}")
    print(f"  Exemples: {numeric_features[:10]}")
    print(f"\nVariables catégoriques: {len(categorical_features)}")
    print(f"  Exemples: {categorical_features[:10]}")
    
    return numeric_features, categorical_features

def visualize_distributions(df, numeric_features, save_plots=False):
    """Visualise les distributions des variables numériques"""
    n_cols = 3
    n_rows = (len(numeric_features) + n_cols - 1) // n_cols
    
    plt.figure(figsize=(20, 5 * n_rows))
    
    for i, col in enumerate(numeric_features, 1):
        plt.subplot(n_rows, n_cols, i)
        data = df[col].dropna()
        
        sns.histplot(data, kde=True, stat="density", alpha=0.7, color='skyblue', linewidth=0)
        
        mean_val = data.mean()
        median_val = data.median()
        
        plt.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Moyenne: {mean_val:.2f}')
        plt.axvline(median_val, color='green', linestyle='-', linewidth=2, label=f'Médiane: {median_val:.2f}')
        
        skewness = stats.skew(data)
        plt.title(f'{col}\nSkewness: {skewness:.2f}', fontsize=12)
        plt.xlabel(col)
        plt.ylabel('Densité')
        plt.legend(fontsize=9)
    
    plt.tight_layout()
    plt.suptitle('Distributions des Variables Numériques (avec Moyenne & Médiane)', fontsize=16, y=1.02)
    if save_plots:
        plt.savefig('output/distributions.png', dpi=300, bbox_inches='tight')
    plt.show()

def correlation_analysis(df, numeric_features, target='Paddy yield(in Kg)', save_plots=False):
    """Analyse de corrélation"""
    corr_matrix = df[numeric_features].corr()
    
    plt.figure(figsize=(14, 12))
    sns.heatmap(corr_matrix,
                annot=True,
                fmt='.2f',
                cmap='coolwarm',
                center=0,
                square=True,
                linewidths=0.5,
                cbar_kws={"shrink": 0.8},
                mask=np.abs(corr_matrix) < 0.01)
    plt.title('Matrice de Corrélation des Variables Numériques',
              fontsize=18, fontweight='bold', pad=20)
    plt.tight_layout()
    if save_plots:
        plt.savefig('output/correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Corrélations avec la cible
    if target in numeric_features:
        print("\n" + "="*80)
        print(f"CORRÉLATION AVEC LA VARIABLE CIBLE : {target}")
        print("="*80)
        
        target_corr = corr_matrix[target].drop(target)
        target_corr_abs = target_corr.abs().sort_values(ascending=False)
        
        print("Top 20 variables les plus corrélées (valeur absolue) :")
        print("-"*80)
        print(target_corr_abs.head(20).round(3))

if __name__ == "__main__":
    # Charger les données
    df = load_data('data/noisy_paddydataset.csv')
    
    # Informations générales
    print("\nINFORMATIONS GÉNÉRALES:")
    print("-" * 30)
    df.info()
    
    # Statistiques descriptives
    print("\nSTATISTIQUES DESCRIPTIVES - VARIABLES NUMÉRIQUES:")
    print("-" * 50)
    print(df.describe())
    
    # Analyse des valeurs manquantes
    missing_df = analyze_missing_values(df)
    
    # Identification des types de variables
    numeric_features, categorical_features = identify_feature_types(df)
    
    # Visualisations
    visualize_distributions(df, numeric_features[:9])  # Afficher les 9 premières
    
    # Analyse de corrélation
    correlation_analysis(df, numeric_features)