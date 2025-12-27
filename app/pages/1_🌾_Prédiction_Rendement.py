import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(page_title="Prédiction Rendement", page_icon="🌾", layout="wide")

# Charger les modèles sauvegardés
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
    
    # Charger le scaler
    try:
        with open("models/scalers/scaler_regression.pkl", 'rb') as f:
            scaler = pickle.load(f)
    except:
        scaler = None
        st.warning("Scaler non trouvé, les prédictions peuvent être moins précises")
    
    return models, scaler

models, scaler = load_regression_models()

# Titre
st.title("🌾 Prédiction du Rendement de Riz")
st.markdown("---")

# Instructions
st.info("""
📊 **Comment ça marche ?**
1. Remplissez les informations sur votre parcelle dans les sections ci-dessous
2. Choisissez le modèle de prédiction
3. Cliquez sur "Prédire le Rendement"
4. Obtenez une estimation précise du rendement en kg
""")

# Formulaire de saisie
st.header("📝 Informations sur la Parcelle")

# Organisation en colonnes
col1, col2 = st.columns(2)

with col1:
    st.subheader("🌱 Parcelle")
    
    hectares = st.number_input("Superficie (hectares)", min_value=0.1, max_value=100.0, value=2.0, step=0.1)
    nursery_area = st.number_input("Surface pépinière (cents)", min_value=0.0, value=50.0, step=5.0)
    seedrate = st.number_input("Taux de semis (kg)", min_value=10.0, max_value=100.0, value=40.0, step=5.0)
    trash = st.number_input("Résidus (bottes)", min_value=0.0, value=20.0, step=5.0)
    
    lp_nursery = st.number_input("LP pépinière (tonnes)", min_value=0.0, value=2.0, step=0.5)
    lp_mainfield = st.number_input("LP champ principal (tonnes)", min_value=0.0, value=5.0, step=0.5)

with col2:
    st.subheader("💊 Intrants")
    
    dap_20days = st.number_input("DAP à 20 jours (kg)", min_value=0.0, value=50.0, step=5.0)
    weed_herbicide = st.number_input("Herbicide 28 jours (ml)", min_value=0.0, value=300.0, step=50.0)
    urea_40days = st.number_input("Urée à 40 jours (kg)", min_value=0.0, value=60.0, step=5.0)
    potash_50days = st.number_input("Potasse à 50 jours (kg)", min_value=0.0, value=40.0, step=5.0)
    micronutrients_70days = st.number_input("Micronutriments à 70 jours (kg)", min_value=0.0, value=10.0, step=1.0)
    pesticide_60days = st.number_input("Pesticide à 60 jours (ml)", min_value=0.0, value=500.0, step=50.0)

st.markdown("---")

# Choix du modèle
st.header("🤖 Sélection du Modèle")
model_choice = st.selectbox(
    "Choisissez le modèle de prédiction",
    list(models.keys()),
    index=0,
    help="Ridge Regression est recommandé pour sa précision et sa robustesse"
)

# Afficher les performances du modèle
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
            # Préparer les données avec les VRAIS noms de colonnes utilisés lors de l'entraînement
            input_data = pd.DataFrame({
                'Hectares ': [hectares],  # Attention à l'espace
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
            
            # Affichage du résultat
            st.success("✅ Prédiction effectuée avec succès !")
            
            # Grande carte de résultat
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
            
            # Interprétation
            st.markdown("### 📊 Interprétation")
            
            if prediction > 4000:
                st.success("""
                🎉 **Excellent rendement prévu !**
                
                Vos conditions sont optimales pour une production élevée. Continuez avec ces pratiques !
                """)
            elif prediction > 2500:
                st.info("""
                ✅ **Rendement acceptable**
                
                Votre rendement est dans la moyenne. Considérez d'optimiser les intrants pour améliorer la production.
                """)
            else:
                st.warning("""
                ⚠️ **Rendement faible prévu**
                
                Plusieurs facteurs peuvent affecter le rendement. Consultez un agronome pour des recommandations personnalisées.
                """)
            
            # Graphique de comparaison
            st.markdown("### 📈 Comparaison avec les Moyennes")
            
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=['Rendement Faible', 'Rendement Moyen', 'Votre Prédiction', 'Rendement Élevé'],
                y=[2000, 3250, prediction, 4500],
                marker_color=['#FF6B6B', '#FFA500', '#4ECDC4', '#95E1D3'],
                text=[2000, 3250, f'{prediction:.0f}', 4500],
                textposition='auto',
            ))
            
            fig.update_layout(
                title="Positionnement de votre rendement",
                yaxis_title="Rendement (kg)",
                height=400,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Recommandations
            st.markdown("### 💡 Recommandations")
            
            recommendations = []
            
            if dap_20days < 40:
                recommendations.append("🌱 Augmentez légèrement l'apport en DAP (recommandé: 40-60 kg)")
            if micronutrients_70days < 8:
                recommendations.append("💊 Complément en micronutriments recommandé (recommandé: 8-12 kg)")
            if seedrate < 30:
                recommendations.append("🌾 Le taux de semis pourrait être augmenté (recommandé: 30-50 kg)")
            if urea_40days < 50:
                recommendations.append("💧 Considérez d'augmenter l'apport en urée (recommandé: 50-70 kg)")
            
            if recommendations:
                for rec in recommendations:
                    st.info(rec)
            else:
                st.success("✅ Vos pratiques culturales sont bien équilibrées !")
            
        except Exception as e:
            st.error(f"❌ Erreur lors de la prédiction : {str(e)}")
            st.info("Vérifiez que tous les champs sont correctement remplis")
            import traceback
            with st.expander("Détails de l'erreur"):
                st.code(traceback.format_exc())
    else:
        st.error("Modèle non disponible")

# Informations complémentaires
with st.expander("ℹ️ À propos des modèles"):
    st.write("""
    **Ridge Regression** (Recommandé)
    - Modèle linéaire régularisé
    - Excellent compromis précision/stabilité
    - R² = 0.925, MAE = 1474 kg
    
    **XGBoost**
    - Modèle de boosting d'arbres
    - Légèrement plus précis
    - R² = 0.926, MAE = 1297 kg
    
    Les modèles ont été entraînés sur 12 features sélectionnées avec validation croisée.
    """)

with st.expander("📖 Guide des intrants"):
    st.write("""
    **DAP (Di-Ammonium Phosphate)** : Apport à 20 jours
    - Dose standard : 40-60 kg/hectare
    
    **Urée** : Apport à 40 jours  
    - Dose standard : 50-70 kg/hectare
    
    **Potasse** : Apport à 50 jours
    - Dose standard : 30-50 kg/hectare
    
    **Micronutriments** : Apport à 70 jours
    - Dose standard : 8-12 kg/hectare
    """)