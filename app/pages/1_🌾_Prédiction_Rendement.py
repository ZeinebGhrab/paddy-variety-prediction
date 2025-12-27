import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
from pathlib import Path
import sys
sys.path.append('app/components')

# Importer le module d'analyse
try:
    from data_analysis import load_and_analyze_data, compare_with_averages
except:
    st.warning("Module d'analyse non trouvé, utilisation des valeurs par défaut")
    load_and_analyze_data = None

st.set_page_config(page_title="Prédiction Rendement", page_icon="🌾", layout="wide")

# Charger les modèles
@st.cache_resource
def load_regression_models():
    """Charge les modèles de régression"""
    models = {}
    model_path = Path("models/regression")
    
    model_files = {
        'Ridge Regression': 'ridge.pkl',
        'Linear Regression': 'linear_regression.pkl',
        'Lasso': 'lasso.pkl',
        'ElasticNet': 'elasticnet.pkl',
        'XGBoost': 'xgboost_reg.pkl'
    }
    
    for name, filename in model_files.items():
        try:
            with open(model_path / filename, 'rb') as f:
                models[name] = pickle.load(f)
        except Exception as e:
            st.warning(f"Impossible de charger {name}: {e}")
    
    try:
        with open("models/scalers/scaler_regression.pkl", 'rb') as f:
            scaler = pickle.load(f)
    except:
        scaler = None
        st.warning("Scaler non trouvé")
    
    return models, scaler

# Charger les données pour analyse
@st.cache_data
def load_dataset_stats():
    """Charge les statistiques du dataset"""
    try:
        df = pd.read_csv("data/cleaned_paddydataset.csv")
        yield_col = 'Paddy yield(in Kg)'
        
        stats = {
            'mean': df[yield_col].mean(),
            'median': df[yield_col].median(),
            'p25': df[yield_col].quantile(0.25),
            'p75': df[yield_col].quantile(0.75),
            'p90': df[yield_col].quantile(0.90),
            'min': df[yield_col].min(),
            'max': df[yield_col].max(),
            'n_parcels': len(df)
        }
        
        # Statistiques par intrant (top 10% vs moyenne)
        top_10_idx = df.nlargest(int(len(df)*0.1), yield_col).index
        
        intrants_stats = {}
        for col in df.select_dtypes(include=[np.number]).columns:
            if col != yield_col:
                intrants_stats[col] = {
                    'mean_all': df[col].mean(),
                    'mean_top10': df.loc[top_10_idx, col].mean(),
                    'optimal': df.loc[top_10_idx, col].median()
                }
        
        return stats, intrants_stats, df
    except Exception as e:
        st.error(f"Erreur chargement stats: {e}")
        return None, None, None

models, scaler = load_regression_models()
dataset_stats, intrants_stats, full_df = load_dataset_stats()

# Titre
st.title("🌾 Prédiction du Rendement de Riz - Analyse Dynamique")
st.markdown("---")

# Afficher stats du dataset
if dataset_stats:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📊 Parcelles", f"{dataset_stats['n_parcels']:,}")
    with col2:
        st.metric("📈 Rendement Moyen", f"{dataset_stats['mean']:.0f} kg/ha")
    with col3:
        st.metric("🏆 Top 10% Moyen", f"{dataset_stats['p90']:.0f} kg/ha")
    with col4:
        st.metric("📉 Rendement Min", f"{dataset_stats['min']:.0f} kg/ha")

# Instructions
st.info(f"""
📊 **Prédiction basée sur {dataset_stats['n_parcels'] if dataset_stats else '...'} parcelles réelles !**

Les recommandations et comparaisons sont calculées dynamiquement à partir des données historiques.
""")

# Formulaire
st.header("📝 Informations sur la Parcelle")

col1, col2 = st.columns(2)

with col1:
    st.subheader("🌱 Parcelle")
    
    hectares = st.number_input("Superficie (hectares)", min_value=0.1, max_value=100.0, value=2.0, step=0.1)
    nursery_area = st.number_input("Surface pépinière (cents)", min_value=0.0, value=50.0, step=5.0)
    seedrate = st.number_input("Taux de semis (kg)", min_value=10.0, max_value=100.0, value=40.0, step=5.0)
    trash = st.number_input("Résidus (bottes)", min_value=0.0, value=20.0, step=5.0)
    
    lp_nursery = st.number_input("LP pépinière (tonnes)", min_value=0.0, value=2.0, step=0.5)
    lp_mainfield = st.number_input("LP champ principal (tonnes)", min_value=0.0, value=5.0, step=0.5)
    
    # Afficher les valeurs optimales du top 10%
    if intrants_stats and 'Hectares ' in intrants_stats:
        with st.expander("💡 Voir valeurs optimales (Top 10%)"):
            st.write(f"**Hectares optimal**: {intrants_stats['Hectares ']['optimal']:.1f}")
            st.write(f"**Semis optimal**: {intrants_stats['Seedrate(in Kg)']['optimal']:.1f} kg")
            st.write(f"**LP Champ optimal**: {intrants_stats['LP_Mainfield(in Tonnes)']['optimal']:.1f} tonnes")

with col2:
    st.subheader("💊 Intrants")
    
    dap_20days = st.number_input("DAP à 20 jours (kg)", min_value=0.0, value=50.0, step=5.0)
    weed_herbicide = st.number_input("Herbicide 28 jours (ml)", min_value=0.0, value=300.0, step=50.0)
    urea_40days = st.number_input("Urée à 40 jours (kg)", min_value=0.0, value=60.0, step=5.0)
    potash_50days = st.number_input("Potasse à 50 jours (kg)", min_value=0.0, value=40.0, step=5.0)
    micronutrients_70days = st.number_input("Micronutriments à 70 jours (kg)", min_value=0.0, value=10.0, step=1.0)
    pesticide_60days = st.number_input("Pesticide à 60 jours (ml)", min_value=0.0, value=500.0, step=50.0)
    
    # Afficher les valeurs optimales
    if intrants_stats:
        with st.expander("💡 Voir intrants optimaux (Top 10%)"):
            st.write(f"**DAP optimal**: {intrants_stats['DAP_20days']['optimal']:.1f} kg")
            st.write(f"**Urée optimale**: {intrants_stats['Urea_40Days']['optimal']:.1f} kg")
            st.write(f"**Potasse optimale**: {intrants_stats['Potassh_50Days']['optimal']:.1f} kg")
            st.write(f"**Micronutriments optimaux**: {intrants_stats['Micronutrients_70Days']['optimal']:.1f} kg")

st.markdown("---")

# Choix du modèle
st.header("🤖 Sélection du Modèle")
model_choice = st.selectbox(
    "Choisissez le modèle de prédiction",
    list(models.keys()),
    index=0
)

# Performances (depuis models_info.json si disponible)
performance_metrics = {
    'Ridge Regression': {'R²': 0.925, 'RMSE': 2401, 'MAE': 1474},
    'Linear Regression': {'R²': 0.925, 'RMSE': 2401, 'MAE': 1473},
    'Lasso': {'R²': 0.925, 'RMSE': 2401, 'MAE': 1473},
    'ElasticNet': {'R²': 0.925, 'RMSE': 2401, 'MAE': 1473},
    'XGBoost': {'R²': 0.926, 'RMSE': 2382, 'MAE': 1297}
}

if model_choice in performance_metrics:
    metrics = performance_metrics[model_choice]
    col1, col2, col3 = st.columns(3)
    col1.metric("R² Score", f"{metrics['R²']:.3f}")
    col2.metric("RMSE", f"{metrics['RMSE']:.0f} kg")
    col3.metric("MAE", f"{metrics['MAE']:.0f} kg")

st.markdown("---")

# Bouton de prédiction
if st.button("🎯 Prédire le Rendement", type="primary", use_container_width=True):
    if model_choice in models:
        try:
            # Préparer les données
            input_data = pd.DataFrame({
                'Hectares ': [hectares],
                'Seedrate(in Kg)': [seedrate],
                'LP_Mainfield(in Tonnes)': [lp_mainfield],
                'Nursery area (Cents)': [nursery_area],
                'LP_nurseryarea(in Tonnes)': [lp_nursery],
                'DAP_20days': [dap_20days],
                'Weed28D_thiobencarb': [weed_herbicide],
                'Urea_40Days': [urea_40days],
                'Potassh_50Days': [potash_50days],
                'Micronutrients_70Days': [micronutrients_70days],
                'Pest_60Day(in ml)': [pesticide_60days],
                'Trash(in bundles)': [trash]
            })
            
            # Normalisation
            if scaler is not None:
                input_scaled = scaler.transform(input_data)
            else:
                input_scaled = input_data.values
            
            # Prédiction
            prediction = models[model_choice].predict(input_scaled)[0]
            
            st.success("✅ Prédiction effectuée avec succès !")
            
            # Grande carte
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        padding: 3rem; border-radius: 20px; text-align: center; margin: 2rem 0;">
                <h1 style="color: white; font-size: 3rem; margin: 0;">
                    {prediction:.0f} kg
                </h1>
                <p style="color: white; font-size: 1.5rem; margin: 1rem 0;">
                    Rendement Estimé
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Positionnement dans le dataset
            if dataset_stats:
                percentile = (np.searchsorted(np.sort(full_df['Paddy yield(in Kg)']), prediction) / len(full_df)) * 100
                
                st.markdown(f"""
                <div style="background: #E3F2FD; padding: 1rem; border-radius: 10px; text-align: center; margin: 1rem 0;">
                    <h4 style="color: #1976D2; margin: 0;">
                        Votre rendement prédit se situe au {percentile:.0f}e percentile du dataset
                    </h4>
                    <p style="color: #1976D2; margin: 0.5rem 0;">
                        Mieux que {percentile:.0f}% des {dataset_stats['n_parcels']} parcelles analysées
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            # Interprétation dynamique
            st.markdown("### 📊 Interprétation")
            
            if dataset_stats:
                if prediction >= dataset_stats['p90']:
                    st.success(f"""
                    🎉 **Rendement exceptionnel !**
                    
                    Votre rendement prédit ({prediction:.0f} kg) est dans le **top 10%** du dataset 
                    (seuil: {dataset_stats['p90']:.0f} kg). Excellentes conditions de culture !
                    """)
                elif prediction >= dataset_stats['p75']:
                    st.success(f"""
                    ✅ **Très bon rendement !**
                    
                    Votre rendement ({prediction:.0f} kg) est au-dessus du 75e percentile 
                    (seuil: {dataset_stats['p75']:.0f} kg). Continuez ces bonnes pratiques !
                    """)
                elif prediction >= dataset_stats['median']:
                    st.info(f"""
                    📊 **Rendement au-dessus de la médiane**
                    
                    Votre rendement ({prediction:.0f} kg) est supérieur à la médiane du dataset 
                    ({dataset_stats['median']:.0f} kg). Bon résultat !
                    """)
                else:
                    st.warning(f"""
                    ⚠️ **Rendement en-dessous de la médiane**
                    
                    Votre rendement prédit ({prediction:.0f} kg) est inférieur à la médiane 
                    ({dataset_stats['median']:.0f} kg). Des optimisations sont possibles.
                    """)
            
            # Graphique de comparaison avec données réelles
            st.markdown("### 📈 Comparaison avec le Dataset")
            
            if dataset_stats:
                fig = go.Figure()
                
                categories = ['Minimum', '25e Percentile', 'Médiane', '75e Percentile', 
                             'Votre Prédiction', 'Top 10%', 'Maximum']
                values = [
                    dataset_stats['min'],
                    dataset_stats['p25'],
                    dataset_stats['median'],
                    dataset_stats['p75'],
                    prediction,
                    dataset_stats['p90'],
                    dataset_stats['max']
                ]
                colors = ['#FF6B6B', '#FFA500', '#FFD700', '#95E1D3', 
                         '#4ECDC4', '#98D8C8', '#66BB6A']
                
                fig.add_trace(go.Bar(
                    x=categories,
                    y=values,
                    marker_color=colors,
                    text=[f'{v:.0f}' for v in values],
                    textposition='auto',
                ))
                
                fig.update_layout(
                    title=f"Positionnement de votre rendement ({prediction:.0f} kg) dans le dataset",
                    yaxis_title="Rendement (kg)",
                    height=450,
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Recommandations basées sur le top 10%
            st.markdown("### 💡 Recommandations (Basées sur le Top 10% du Dataset)")
            
            if intrants_stats:
                recommendations = []
                
                # Comparer avec le top 10%
                user_intrants = {
                    'DAP_20days': dap_20days,
                    'Urea_40Days': urea_40days,
                    'Potassh_50Days': potash_50days,
                    'Micronutrients_70Days': micronutrients_70days,
                    'Seedrate(in Kg)': seedrate,
                    'LP_Mainfield(in Tonnes)': lp_mainfield
                }
                
                for key, user_val in user_intrants.items():
                    if key in intrants_stats:
                        optimal_val = intrants_stats[key]['optimal']
                        mean_top10 = intrants_stats[key]['mean_top10']
                        
                        diff_percent = abs(user_val - optimal_val) / optimal_val * 100
                        
                        if diff_percent > 25:
                            label = key.replace('_', ' ').replace('(in Kg)', '').replace('(in Tonnes)', '')
                            if user_val < optimal_val:
                                recommendations.append({
                                    'type': 'warning',
                                    'text': f"⬆️ **{label}**: Augmentez de {user_val:.1f} vers {optimal_val:.1f} (valeur du top 10%)"
                                })
                            else:
                                recommendations.append({
                                    'type': 'info',
                                    'text': f"⬇️ **{label}**: Réduisez de {user_val:.1f} vers {optimal_val:.1f} (optimal)"
                                })
                
                if recommendations:
                    st.write("**Suggestions d'optimisation basées sur les meilleures parcelles:**")
                    for rec in recommendations:
                        if rec['type'] == 'warning':
                            st.warning(rec['text'])
                        else:
                            st.info(rec['text'])
                else:
                    st.success("✅ Vos intrants sont alignés avec les meilleures pratiques du dataset !")
            
            # Tableau comparatif
            with st.expander("📋 Comparaison Détaillée de vos Intrants vs Top 10%"):
                if intrants_stats:
                    comparison = []
                    for key, user_val in user_intrants.items():
                        if key in intrants_stats:
                            comparison.append({
                                'Intrant': key.replace('_', ' '),
                                'Votre Valeur': f"{user_val:.1f}",
                                'Moyenne Globale': f"{intrants_stats[key]['mean_all']:.1f}",
                                'Moyenne Top 10%': f"{intrants_stats[key]['mean_top10']:.1f}",
                                'Optimal (Médiane Top 10%)': f"{intrants_stats[key]['optimal']:.1f}"
                            })
                    
                    comp_df = pd.DataFrame(comparison)
                    st.dataframe(comp_df, use_container_width=True, hide_index=True)
            
        except Exception as e:
            st.error(f"❌ Erreur lors de la prédiction : {str(e)}")
            import traceback
            with st.expander("Détails de l'erreur"):
                st.code(traceback.format_exc())
    else:
        st.error("Modèle non disponible")

# Informations complémentaires
with st.expander("ℹ️ À propos des modèles et données"):
    st.write(f"""
    **Modèles entraînés sur {dataset_stats['n_parcels'] if dataset_stats else '...'} parcelles réelles**
    
    - Les recommandations sont basées sur l'analyse du **top 10%** des rendements
    - Les comparaisons utilisent les percentiles réels du dataset
    - Les valeurs optimales sont calculées dynamiquement
    
    **Ridge Regression** (Recommandé)
    - R² = {performance_metrics['Ridge Regression']['R²']:.3f}
    - RMSE = {performance_metrics['Ridge Regression']['RMSE']:.0f} kg
    """)

with st.expander("📊 Distribution des Rendements dans le Dataset"):
    if full_df is not None:
        import plotly.express as px
        fig = px.histogram(
            full_df, 
            x='Paddy yield(in Kg)',
            nbins=50,
            title="Distribution des rendements dans le dataset"
        )
        fig.add_vline(x=dataset_stats['median'], line_dash="dash", 
                     annotation_text="Médiane", line_color="red")
        st.plotly_chart(fig, use_container_width=True)