"""
Module d'analyse dynamique des données agricoles
Extrait les statistiques et recommandations depuis le dataset
"""
import pandas as pd
import numpy as np
from pathlib import Path
import streamlit as st

@st.cache_data
def load_and_analyze_data():
    """
    Charge et analyse le dataset pour extraire les informations sur chaque variété
    
    Returns:
        dict: Informations complètes sur chaque variété basées sur les données réelles
    """
    try:
        df = pd.read_csv("data/cleaned_paddydataset.csv")
        
        # Trouver la colonne Variety
        variety_col = None
        for col in df.columns:
            if 'variety' in col.lower():
                variety_col = col
                break
        
        if variety_col is None:
            return None
            
        # Trouver la colonne de rendement
        yield_col = 'Paddy yield(in Kg)'
        if yield_col not in df.columns:
            yield_col = [col for col in df.columns if 'yield' in col.lower()][0]
        
        # Analyser chaque variété
        varieties_info = {}
        
        for variety in df[variety_col].unique():
            variety_data = df[df[variety_col] == variety]
            
            # Statistiques de rendement
            yield_stats = {
                'mean': variety_data[yield_col].mean(),
                'median': variety_data[yield_col].median(),
                'min': variety_data[yield_col].min(),
                'max': variety_data[yield_col].max(),
                'std': variety_data[yield_col].std(),
                'q25': variety_data[yield_col].quantile(0.25),
                'q75': variety_data[yield_col].quantile(0.75)
            }
            
            # Sol préféré (mode)
            soil_col = [col for col in df.columns if 'soil' in col.lower()]
            preferred_soil = variety_data[soil_col[0]].mode()[0] if soil_col else 'non défini'
            
            # Type de pépinière préféré
            nursery_col = [col for col in df.columns if 'nursery' in col.lower() and 'area' not in col.lower()]
            preferred_nursery = variety_data[nursery_col[0]].mode()[0] if nursery_col else 'non défini'
            
            # Moyennes des intrants pour cette variété
            intrants = {}
            for col in variety_data.select_dtypes(include=[np.number]).columns:
                if col != yield_col:
                    intrants[col] = {
                        'mean': variety_data[col].mean(),
                        'optimal': variety_data.nlargest(int(len(variety_data)*0.1), yield_col)[col].mean()
                    }
            
            # Nombre de parcelles
            n_parcels = len(variety_data)
            
            # Pourcentage du total
            percent_total = (n_parcels / len(df)) * 100
            
            varieties_info[variety] = {
                'yield_stats': yield_stats,
                'preferred_soil': preferred_soil,
                'preferred_nursery': preferred_nursery,
                'intrants': intrants,
                'n_parcels': n_parcels,
                'percent_total': percent_total,
                'raw_data': variety_data
            }
        
        return varieties_info, df, variety_col, yield_col
        
    except Exception as e:
        st.error(f"Erreur lors de l'analyse : {e}")
        return None

def get_variety_characteristics(variety_name, variety_info):
    """
    Génère la liste des caractéristiques d'une variété basée sur les données réelles
    
    Args:
        variety_name: Nom de la variété
        variety_info: Informations sur la variété
    
    Returns:
        list: Liste des caractéristiques
    """
    info = variety_info[variety_name]
    characteristics = []
    
    # Calculer les seuils dynamiquement depuis TOUTES les variétés
    all_means = [v['yield_stats']['mean'] for v in variety_info.values()]
    threshold_high = np.percentile(all_means, 66.67)  # Top 33%
    threshold_low = np.percentile(all_means, 33.33)   # Bottom 33%
    
    # Rendement avec classification dynamique
    yield_mean = info['yield_stats']['mean']
    if yield_mean >= threshold_high:
        characteristics.append(f"✓ Rendement élevé: {yield_mean:.0f} kg/ha (top 33% des variétés)")
    elif yield_mean >= threshold_low:
        characteristics.append(f"✓ Rendement moyen: {yield_mean:.0f} kg/ha (33-66e percentile)")
    else:
        characteristics.append(f"✓ Rendement modéré: {yield_mean:.0f} kg/ha (bottom 33%)")
    
    # Plage de rendement
    yield_min = info['yield_stats']['q25']
    yield_max = info['yield_stats']['q75']
    characteristics.append(f"✓ Plage typique: {yield_min:.0f}-{yield_max:.0f} kg/ha")
    
    # Sol préféré
    characteristics.append(f"✓ Sol optimal: {info['preferred_soil'].capitalize()}")
    
    # Pépinière
    characteristics.append(f"✓ Pépinière: {info['preferred_nursery'].capitalize()}")
    
    # Stabilité (basée sur std)
    cv = info['yield_stats']['std'] / info['yield_stats']['mean'] * 100
    if cv < 20:
        characteristics.append("✓ Rendement stable et prévisible")
    elif cv < 30:
        characteristics.append("✓ Rendement modérément variable")
    else:
        characteristics.append("✓ Rendement variable selon conditions")
    
    return characteristics

def get_variety_description(variety_name, variety_info):
    """
    Génère une description d'une variété basée sur les données
    
    Args:
        variety_name: Nom de la variété
        variety_info: Informations sur la variété
    
    Returns:
        str: Description de la variété
    """
    info = variety_info[variety_name]
    
    soil = info['preferred_soil']
    yield_level = info['yield_stats']['mean']
    percent = info['percent_total']
    
    if yield_level > 4000:
        yield_desc = "rendement élevé"
    elif yield_level > 3000:
        yield_desc = "rendement moyen"
    else:
        yield_desc = "rendement modéré"
    
    description = f"Variété à {yield_desc}, préfère les sols {soil}. "
    description += f"Représente {percent:.1f}% des cultures dans le dataset."
    
    return description

def get_recommendations_for_inputs(user_inputs, variety_info, top_variety):
    """
    Génère des recommandations personnalisées basées sur les données
    
    Args:
        user_inputs: Dictionnaire des intrants de l'utilisateur
        variety_info: Informations sur toutes les variétés
        top_variety: Variété recommandée
    
    Returns:
        list: Liste de recommandations
    """
    recommendations = []
    info = variety_info[top_variety]
    
    # Comparer les intrants de l'utilisateur avec les optimaux
    intrants_keys_map = {
        'dap_20days': 'DAP_20days',
        'urea_40days': 'Urea_40Days',
        'potash_50days': 'Potassh_50Days',
        'micronutrients_70days': 'Micronutrients_70Days',
        'hectares': 'Hectares ',
        'seedrate': 'Seedrate(in Kg)'
    }
    
    for user_key, data_key in intrants_keys_map.items():
        if user_key in user_inputs and data_key in info['intrants']:
            user_val = user_inputs[user_key]
            optimal_val = info['intrants'][data_key]['optimal']
            mean_val = info['intrants'][data_key]['mean']
            
            # Recommandation si valeur trop éloignée de l'optimal
            diff_percent = abs(user_val - optimal_val) / optimal_val * 100
            
            if diff_percent > 30:
                if user_val < optimal_val:
                    recommendations.append(
                        f"💡 Augmentez {user_key.replace('_', ' ').title()} "
                        f"(actuel: {user_val:.1f}, optimal pour {top_variety}: {optimal_val:.1f})"
                    )
                else:
                    recommendations.append(
                        f"⚠️ Réduisez {user_key.replace('_', ' ').title()} "
                        f"(actuel: {user_val:.1f}, optimal pour {top_variety}: {optimal_val:.1f})"
                    )
    
    # Recommandation sur le sol
    if 'soil_type' in user_inputs:
        preferred_soil = info['preferred_soil']
        if user_inputs['soil_type'] == preferred_soil:
            recommendations.append(f"✅ Excellent ! Le sol {preferred_soil} est idéal pour {top_variety}")
        else:
            recommendations.append(
                f"ℹ️ Note: {top_variety} préfère les sols {preferred_soil}, "
                f"mais peut s'adapter à votre sol {user_inputs['soil_type']}"
            )
    
    # Si pas de recommandations spécifiques
    if not recommendations:
        recommendations.append(
            f"✅ Vos intrants sont bien équilibrés pour cultiver {top_variety} !"
        )
    
    return recommendations

def compare_with_averages(predicted_yield, variety_info):
    """
    Compare le rendement prédit avec les moyennes du dataset
    
    Args:
        predicted_yield: Rendement prédit
        variety_info: Informations sur les variétés
    
    Returns:
        dict: Informations de comparaison
    """
    # Calculer les percentiles globaux
    all_yields = []
    for info in variety_info.values():
        all_yields.extend(info['raw_data']['Paddy yield(in Kg)'].tolist())
    
    all_yields = np.array(all_yields)
    
    return {
        'faible': np.percentile(all_yields, 25),
        'moyen': np.percentile(all_yields, 50),
        'eleve': np.percentile(all_yields, 75),
        'excellent': np.percentile(all_yields, 90),
        'percentile': np.searchsorted(np.sort(all_yields), predicted_yield) / len(all_yields) * 100
    }

def get_optimal_practices_for_variety(variety_name, variety_info, n_top=10):
    """
    Identifie les pratiques optimales pour une variété (top 10% rendements)
    
    Args:
        variety_name: Nom de la variété
        variety_info: Informations sur la variété
        n_top: Pourcentage du top à considérer
    
    Returns:
        dict: Pratiques optimales
    """
    info = variety_info[variety_name]
    data = info['raw_data']
    yield_col = 'Paddy yield(in Kg)'
    
    # Top parcelles
    top_parcels = data.nlargest(int(len(data) * n_top / 100), yield_col)
    
    optimal_practices = {}
    
    for col in top_parcels.select_dtypes(include=[np.number]).columns:
        if col != yield_col:
            optimal_practices[col] = {
                'mean': top_parcels[col].mean(),
                'median': top_parcels[col].median(),
                'min': top_parcels[col].min(),
                'max': top_parcels[col].max()
            }
    
    return optimal_practices