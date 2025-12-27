import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
from pathlib import Path
import sys
sys.path.append('app/components')

# Importer le module d'analyse dynamique
try:
    from data_analysis import (
        load_and_analyze_data,
        get_variety_characteristics,
        get_variety_description,
        get_recommendations_for_inputs,
        get_optimal_practices_for_variety
    )
except:
    # Si le module n'est pas trouvé, créer une version inline
    st.error("Module d'analyse non trouvé. Assurez-vous que data_analysis.py est dans app/components/")
    st.stop()

st.set_page_config(page_title="Recommandation Variété", page_icon="🌱", layout="wide")

# Charger les modèles
@st.cache_resource
def load_classification_models():
    """Charge les modèles de classification"""
    models = {}
    model_path = Path("models/classification")
    
    model_files = {
        'XGBoost': 'xgboost_clf.pkl',
        'Random Forest': 'random_forest.pkl',
        'Logistic Regression': 'logistic_regression.pkl',
        'KNN': 'knn.pkl',
        'Decision Tree': 'decision_tree.pkl'
    }
    
    for name, filename in model_files.items():
        try:
            with open(model_path / filename, 'rb') as f:
                models[name] = pickle.load(f)
        except Exception as e:
            st.warning(f"Impossible de charger {name}: {e}")
    
    try:
        with open("models/scalers/scaler_classification.pkl", 'rb') as f:
            scaler = pickle.load(f)
    except:
        scaler = None
    
    return models, scaler

@st.cache_data
def load_training_columns():
    """Charge les colonnes du dataset d'entraînement"""
    try:
        df = pd.read_csv("data/cleaned_paddydataset.csv")
        variety_col = None
        for col in df.columns:
            if 'variety' in col.lower():
                variety_col = col
                break
        
        if variety_col:
            X = df.drop(variety_col, axis=1)
            categorical_features = X.select_dtypes(include=['object']).columns.tolist()
            X_encoded = pd.get_dummies(X, columns=categorical_features, drop_first=False, dtype=int)
            return X_encoded.columns.tolist(), df
        return None, None
    except:
        return None, None

# Charger les données et analyses
models, scaler = load_classification_models()
training_columns, training_df = load_training_columns()
analysis_result = load_and_analyze_data()

if analysis_result is None:
    st.error("❌ Impossible de charger et analyser les données")
    st.stop()

varieties_info, full_df, variety_col, yield_col = analysis_result

# Mapping des variétés (ordre alphabétique)
variety_names_list = sorted(varieties_info.keys())
VARIETY_NAMES = {i: name for i, name in enumerate(variety_names_list)}

# Emojis et couleurs pour l'affichage
VARIETY_COLORS = {
    variety_names_list[0]: '#FF6B6B',
    variety_names_list[1]: '#4ECDC4',
    variety_names_list[2]: '#95E1D3' if len(variety_names_list) > 2 else '#FFA500'
}

VARIETY_EMOJIS = {
    variety_names_list[0]: '🌾',
    variety_names_list[1]: '🌿',
    variety_names_list[2]: '⭐' if len(variety_names_list) > 2 else '🌱'
}

# Titre
st.title("🌱 Recommandation de Variété de Riz - Analyse Dynamique")
st.markdown("---")

# Info dynamique sur le dataset
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("📊 Parcelles Analysées", f"{len(full_df):,}")
with col2:
    st.metric("🌾 Variétés Disponibles", len(varieties_info))
with col3:
    avg_yield = full_df[yield_col].mean()
    st.metric("📈 Rendement Moyen Global", f"{avg_yield:.0f} kg/ha")

st.info(f"""
🎯 **Trouvez la variété parfaite basée sur {len(full_df):,} parcelles réelles !**

Les variétés disponibles dans notre dataset :
{', '.join([f'**{v}** ({varieties_info[v]["n_parcels"]} parcelles)' for v in variety_names_list])}

Toutes les recommandations sont extraites des données réelles de culture.
""")

if training_df is None or training_columns is None:
    st.error("❌ Impossible de charger les données d'entraînement.")
    st.stop()

# Formulaire
st.header("📝 Informations sur votre Parcelle")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("🌍 Localisation & Sol")
    agriblock = st.selectbox("Bloc Agricole", [f"block_{i}" for i in range(1, 11)])
    
    # Sols disponibles dans le dataset
    soil_types = full_df[[c for c in full_df.columns if 'soil' in c.lower()][0]].unique()
    soil_type = st.selectbox("Type de Sol", sorted(soil_types))
    
    nursery_types = full_df[[c for c in full_df.columns if 'nursery' in c.lower() and 'area' not in c.lower()][0]].unique()
    nursery_type = st.selectbox("Type de pépinière", sorted(nursery_types))

with col2:
    st.subheader("📏 Parcelle & Intrants Basiques")
    hectares = st.number_input("Hectares", min_value=0.1, max_value=100.0, value=2.0, step=0.1)
    nursery_area = st.number_input("Surface pépinière (cents)", min_value=0.0, value=50.0, step=5.0)
    seedrate = st.number_input("Taux de semis (kg)", min_value=10.0, max_value=100.0, value=40.0, step=5.0)
    lp_mainfield = st.number_input("LP champ principal (tonnes)", min_value=0.0, value=5.0, step=0.5)
    lp_nursery = st.number_input("LP pépinière (tonnes)", min_value=0.0, value=2.0, step=0.5)

with col3:
    st.subheader("💊 Engrais & Pesticides")
    dap_20days = st.number_input("DAP 20 jours (kg)", min_value=0.0, value=50.0, step=5.0)
    urea_40days = st.number_input("Urée 40 jours (kg)", min_value=0.0, value=60.0, step=5.0)
    potash_50days = st.number_input("Potasse 50 jours (kg)", min_value=0.0, value=40.0, step=5.0)
    micronutrients_70days = st.number_input("Micronutriments 70 jours (kg)", min_value=0.0, value=10.0, step=1.0)
    weed_herbicide = st.number_input("Herbicide 28 jours (ml)", min_value=0.0, value=300.0, step=50.0)
    pesticide_60days = st.number_input("Pesticide 60 jours (ml)", min_value=0.0, value=500.0, step=50.0)

st.markdown("---")

# Sélection du modèle
st.header("🤖 Modèle de Recommandation")
model_choice = st.selectbox(
    "Choisissez le modèle",
    list(models.keys()),
    index=0
)

# Performances
performance_metrics = {
    'XGBoost': {'Accuracy': 0.887, 'F1-Score': 0.887, 'ROC-AUC': 0.956},
    'Random Forest': {'Accuracy': 0.805, 'F1-Score': 0.805, 'ROC-AUC': 0.920},
    'Logistic Regression': {'Accuracy': 0.656, 'F1-Score': 0.608, 'ROC-AUC': 0.812},
    'KNN': {'Accuracy': 0.396, 'F1-Score': 0.401, 'ROC-AUC': 0.559},
    'Decision Tree': {'Accuracy': 0.634, 'F1-Score': 0.630, 'ROC-AUC': 0.814}
}

if model_choice in performance_metrics:
    metrics = performance_metrics[model_choice]
    col1, col2, col3 = st.columns(3)
    col1.metric("Précision", f"{metrics['Accuracy']:.1%}")
    col2.metric("F1-Score", f"{metrics['F1-Score']:.1%}")
    col3.metric("ROC-AUC", f"{metrics['ROC-AUC']:.3f}")

st.markdown("---")

# Bouton de recommandation
if st.button("🎯 Obtenir une Recommandation", type="primary", use_container_width=True):
    if model_choice in models:
        try:
            # Préparer les données
            median_values = {}
            for col in training_df.columns:
                if col.lower() not in ['variety', 'paddy yield(in kg)']:
                    if training_df[col].dtype in [np.float64, np.int64]:
                        median_values[col] = training_df[col].median()
                    else:
                        median_values[col] = training_df[col].mode()[0] if len(training_df[col].mode()) > 0 else training_df[col].iloc[0]
            
            user_input = {
                'Hectares ': hectares,
                'Agriblock': agriblock,
                'Soil Types': soil_type,
                'Seedrate(in Kg)': seedrate,
                'LP_Mainfield(in Tonnes)': lp_mainfield,
                'Nursery': nursery_type,
                'Nursery area (Cents)': nursery_area,
                'LP_nurseryarea(in Tonnes)': lp_nursery,
                'DAP_20days': dap_20days,
                'Weed28D_thiobencarb': weed_herbicide,
                'Urea_40Days': urea_40days,
                'Potassh_50Days': potash_50days,
                'Micronutrients_70Days': micronutrients_70days,
                'Pest_60Day(in ml)': pesticide_60days
            }
            
            median_values.update(user_input)
            input_df = pd.DataFrame([median_values])
            
            categorical_features = input_df.select_dtypes(include=['object']).columns.tolist()
            if categorical_features:
                input_encoded = pd.get_dummies(input_df, columns=categorical_features, drop_first=False, dtype=int)
            else:
                input_encoded = input_df
            
            for col in training_columns:
                if col not in input_encoded.columns:
                    input_encoded[col] = 0
            
            input_encoded = input_encoded[training_columns]
            
            if scaler is not None:
                input_scaled = scaler.transform(input_encoded)
            else:
                input_scaled = input_encoded.values
            
            # Prédiction
            prediction = models[model_choice].predict(input_scaled)[0]
            
            if hasattr(models[model_choice], 'predict_proba'):
                probas = models[model_choice].predict_proba(input_scaled)[0]
            else:
                probas = np.array([1/len(VARIETY_NAMES)] * len(VARIETY_NAMES))
            
            recommended_variety = VARIETY_NAMES[prediction]
            
            # Obtenir les infos dynamiques
            variety_description = get_variety_description(recommended_variety, varieties_info)
            variety_characteristics = get_variety_characteristics(recommended_variety, varieties_info)
            variety_color = VARIETY_COLORS.get(recommended_variety, '#4ECDC4')
            variety_emoji = VARIETY_EMOJIS.get(recommended_variety, '🌾')
            
            st.success("✅ Recommandation générée avec succès basée sur les données réelles !")
            
            # Carte de recommandation
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, {variety_color} 0%, #764ba2 100%); 
                        padding: 3rem; border-radius: 20px; text-align: center; margin: 2rem 0;">
                <h1 style="color: white; font-size: 4rem; margin: 0;">
                    {variety_emoji} {recommended_variety}
                </h1>
                <p style="color: white; font-size: 1.5rem; margin: 1rem 0;">
                    Variété Recommandée
                </p>
                <p style="color: white; font-size: 1.2rem; opacity: 0.9;">
                    {variety_description}
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Niveau de confiance
            confidence = probas[prediction] * 100
            st.markdown(f"""
            <div style="background: #E8F5E9; padding: 1rem; border-radius: 10px; text-align: center;">
                <h3 style="color: #2E7D32; margin: 0;">
                    Niveau de Confiance: {confidence:.1f}%
                </h3>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Caractéristiques et probabilités
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("### 🌾 Caractéristiques (Données Réelles)")
                for char in variety_characteristics:
                    st.markdown(f"**{char}**")
                
                # Statistiques supplémentaires
                st.markdown("#### 📊 Statistiques du Dataset")
                info = varieties_info[recommended_variety]
                st.write(f"- Nombre de parcelles: **{info['n_parcels']}**")
                st.write(f"- Rendement médian: **{info['yield_stats']['median']:.0f} kg/ha**")
                st.write(f"- Rendement max observé: **{info['yield_stats']['max']:.0f} kg/ha**")
            
            with col2:
                st.markdown("### 📊 Distribution des Probabilités")
                
                fig = go.Figure(data=[
                    go.Bar(
                        x=list(VARIETY_NAMES.values()),
                        y=probas * 100,
                        marker_color=[VARIETY_COLORS.get(v, '#4ECDC4') for v in VARIETY_NAMES.values()],
                        text=[f"{p:.1f}%" for p in probas * 100],
                        textposition='auto',
                    )
                ])
                
                fig.update_layout(
                    title="Probabilités pour chaque variété",
                    yaxis_title="Probabilité (%)",
                    height=300,
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            
            # Comparaison basée sur données réelles
            st.markdown("### 📊 Comparaison des Variétés (Données Réelles)")
            
            comparison_data = []
            for variety in VARIETY_NAMES.values():
                info = varieties_info[variety]
                comparison_data.append({
                    'Variété': variety,
                    'Probabilité': f"{probas[list(VARIETY_NAMES.values()).index(variety)] * 100:.1f}%",
                    'Rendement Moyen': f"{info['yield_stats']['mean']:.0f} kg/ha",
                    'Rendement Médian': f"{info['yield_stats']['median']:.0f} kg/ha",
                    'Sol Préféré': info['preferred_soil'].capitalize(),
                    'Parcelles': info['n_parcels']
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True, hide_index=True)
            
            # Recommandations personnalisées
            st.markdown("### 💡 Conseils Personnalisés (Basés sur le Top 10%)")
            
            user_inputs_dict = {
                'dap_20days': dap_20days,
                'urea_40days': urea_40days,
                'potash_50days': potash_50days,
                'micronutrients_70days': micronutrients_70days,
                'hectares': hectares,
                'seedrate': seedrate,
                'soil_type': soil_type
            }
            
            recommendations = get_recommendations_for_inputs(
                user_inputs_dict, 
                varieties_info, 
                recommended_variety
            )
            
            for rec in recommendations:
                if rec.startswith('✅'):
                    st.success(rec)
                elif rec.startswith('⚠️'):
                    st.warning(rec)
                else:
                    st.info(rec)
            
            # Pratiques optimales
            with st.expander("📈 Voir les Pratiques Optimales du Top 10%"):
                optimal = get_optimal_practices_for_variety(recommended_variety, varieties_info, n_top=10)
                
                st.write(f"**Intrants moyens des 10% meilleures parcelles de {recommended_variety}:**")
                
                for key, values in optimal.items():
                    if 'DAP' in key or 'Urea' in key or 'Potassh' in key or 'Micronutrients' in key:
                        st.write(f"- **{key}**: {values['median']:.1f} (plage: {values['min']:.0f}-{values['max']:.0f})")
            
        except Exception as e:
            st.error(f"❌ Erreur : {str(e)}")
            import traceback
            with st.expander("Détails de l'erreur"):
                st.code(traceback.format_exc())
    else:
        st.error("Modèle non disponible")

# Informations complémentaires dynamiques
with st.expander("📊 Distribution des Variétés dans le Dataset"):
    fig = go.Figure(data=[
        go.Pie(
            labels=list(varieties_info.keys()),
            values=[info['n_parcels'] for info in varieties_info.values()],
            marker_colors=[VARIETY_COLORS.get(v, '#4ECDC4') for v in varieties_info.keys()]
        )
    ])
    fig.update_layout(title="Répartition des parcelles par variété")
    st.plotly_chart(fig, use_container_width=True)

with st.expander("📈 Performance des modèles"):
    st.write("""
    Les modèles ont été entraînés sur les données réelles de votre dataset.
    
    **XGBoost** (Recommandé)
    - Le plus précis pour capturer les interactions complexes
    - Robuste aux données manquantes
    
    Les recommandations et caractéristiques sont **extraites dynamiquement** 
    du dataset, reflétant les conditions réelles de culture.
    """)