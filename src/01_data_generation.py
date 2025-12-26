import pandas as pd
import numpy as np

def add_noise(df, uppercase_prob=0.10, missing_prob=0.05):
    """Ajoute du bruit aux données"""
    df_noisy = df.copy()
    
    # Pour chaque colonne de type string (textuelle)
    for col in df_noisy.select_dtypes(include=['object']).columns:
        # Appliquer aléatoirement la majuscule sur 10% des valeurs
        mask = np.random.rand(len(df_noisy)) < uppercase_prob
        df_noisy.loc[mask, col] = df_noisy.loc[mask, col].str.upper()
    
    # Pour toutes les colonnes : supprimer aléatoirement 5% des valeurs
    for col in df_noisy.columns:
        mask = np.random.rand(len(df_noisy)) < missing_prob
        df_noisy.loc[mask, col] = np.nan
    
    return df_noisy

if __name__ == "__main__":
    # Charger le fichier original
    df = pd.read_csv('data/paddydataset.csv', sep=',', encoding='utf-8', low_memory=False)
    
    # Appliquer le bruit
    noisy_df = add_noise(df, uppercase_prob=0.10, missing_prob=0.08)
    
    # Enregistrer
    noisy_df.to_csv('data/noisy_paddydataset.csv', index=False, encoding='utf-8')
    
    print("Le dataset noisy a été enregistré avec succès dans : data/noisy_paddydataset.csv")
    print(f"Nombre de lignes : {len(noisy_df)}")
    print(f"Pourcentage de valeurs manquantes total : {noisy_df.isna().mean().mean():.2%}")