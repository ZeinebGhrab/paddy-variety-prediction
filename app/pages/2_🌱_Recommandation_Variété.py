import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
from pathlib import Path

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
    
    # Charger le scaler
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
        # Trouver la colonne Variety
        variety_col = None
        for col in df.columns:
            if 'variety' in col.lower():
                variety_col = col
                break
        
        if variety_col:
            # Retourner X (sans Variety) et les colonnes après encodage
            X = df.drop(variety_col, axis=1)
            categorical_features = X.select_dtypes(include=['object']).columns.tolist()
            X_encoded = pd.get_dummies(X, columns=categorical_features, drop_first=False, dtype=int)
            return X_encoded.columns.tolist(), df
        return None, None
    except:
        return None, None

models, scaler = load_classification_models()
training_columns, training_df = load_training_columns()

# Mapping des variétés
VARIETY_NAMES = {
    0: 'CO_43',
    1: 'Ponmani', 
    2: 'Delux Ponni'
}

VARIETY_INFO = {
    'CO_43': {
        'emoji': '🌾',
        'description': 'Variété résistante, adaptée aux sols alluviaux et conditions sèches',
        'characteristics': [
            '✓ Résistance à la sécheresse',
            '✓ Cycle de 130-135 jours',
            '✓ Rendement moyen: 3500-4000 kg/ha',
            '✓ Grains moyens à longs'
        ],
        'color': '#FF6B6B'
    },
    'Ponmani': {
        'emoji': '🌿',
        'description': 'Variété premium, préfère les sols argileux humides',
        'characteristics': [
            '✓ Qualité de grain excellente',
            '✓ Cycle de 145-150 jours',
            '✓ Rendement élevé: 4000-4500 kg/ha',
            '✓ Préfère humidité élevée'
        ],
        'color': '#4ECDC4'
    },
    'Delux Ponni': {
        'emoji': '⭐',
        'description': 'Variété polyvalente, haut rendement',
        'characteristics': [
            '✓ Très bon rendement',
            '✓ Cycle de 135-140 jours',
            '✓ Rendement: 4200-4800 kg/ha',
            '✓ Adaptable à différents sols'
        ],
        'color': '#95E1D3'
    }
}

# Titre
st.title("🌱 Recommandation de Variété de Riz")
st.markdown("---")

# Instructions
st.info("""
🎯 **Trouvez la variété parfaite pour votre parcelle !**

Cette page vous aide à choisir parmi 3 variétés de riz :
- **CO_43** : Résistant et fiable
- **Ponmani** : Qualité premium
- **Delux Ponni** : Rendement maximal

Remplissez les informations ci-dessous pour obtenir une recommandation personnalisée.
""")

if training_df is None or training_columns is None:
    st.error("❌ Impossible de charger les données d'entraînement. Vérifiez que le fichier `data/cleaned_paddydataset.csv` existe.")
    st.stop()

# Formulaire simplifié avec les VRAIES colonnes
st.header("📝 Informations sur votre Parcelle")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("🌍 Localisation & Sol")
    agriblock = st.selectbox("Bloc Agricole", [f"block_{i}" for i in range(1, 11)])
    soil_type = st.selectbox("Type de Sol", 
                             ["alluvial", "clay", "loamy", "sandy"],
                             help="Alluvial: Léger, Clay: Argileux, Loamy: Limoneux, Sandy: Sableux")
    nursery_type = st.selectbox("Type de pépinière", ["wet", "dry"])

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
    index=0,
    help="XGBoost offre la meilleure précision (88.7%)"
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
            # Créer un DataFrame avec les médianes pour toutes les colonnes
            median_values = {}
            for col in training_df.columns:
                if col.lower() not in ['variety', 'paddy yield(in kg)']:
                    if training_df[col].dtype in [np.float64, np.int64]:
                        median_values[col] = training_df[col].median()
                    else:
                        median_values[col] = training_df[col].mode()[0] if len(training_df[col].mode()) > 0 else training_df[col].iloc[0]
            
            # Remplacer par les valeurs saisies (avec les VRAIS noms de colonnes)
            user_input = {
                'Hectares ': hectares,  # Attention à l'espace !
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
            
            # Fusionner avec les médianes
            median_values.update(user_input)
            
            # Créer DataFrame
            input_df = pd.DataFrame([median_values])
            
            # Encodage one-hot des variables catégorielles
            categorical_features = input_df.select_dtypes(include=['object']).columns.tolist()
            if categorical_features:
                input_encoded = pd.get_dummies(input_df, columns=categorical_features, drop_first=False, dtype=int)
            else:
                input_encoded = input_df
            
            # Ajouter les colonnes manquantes avec des 0
            for col in training_columns:
                if col not in input_encoded.columns:
                    input_encoded[col] = 0
            
            # Garder uniquement les colonnes du modèle dans le bon ordre
            input_encoded = input_encoded[training_columns]
            
            # Normalisation
            if scaler is not None:
                input_scaled = scaler.transform(input_encoded)
            else:
                input_scaled = input_encoded.values
            
            # Prédiction
            prediction = models[model_choice].predict(input_scaled)[0]
            
            # Probabilités
            if hasattr(models[model_choice], 'predict_proba'):
                probas = models[model_choice].predict_proba(input_scaled)[0]
            else:
                probas = np.array([0.33, 0.33, 0.34])
            
            recommended_variety = VARIETY_NAMES[prediction]
            variety_info = VARIETY_INFO[recommended_variety]
            
            # Affichage du résultat
            st.success("✅ Recommandation générée avec succès !")
            
            # Grande carte de recommandation
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, {variety_info['color']} 0%, #764ba2 100%); 
                        padding: 3rem; border-radius: 20px; text-align: center; margin: 2rem 0;">
                <h1 style="color: white; font-size: 4rem; margin: 0;">
                    {variety_info['emoji']} {recommended_variety}
                </h1>
                <p style="color: white; font-size: 1.5rem; margin: 1rem 0;">
                    Variété Recommandée
                </p>
                <p style="color: white; font-size: 1.2rem; opacity: 0.9;">
                    {variety_info['description']}
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
                st.markdown("### 🌾 Caractéristiques")
                for char in variety_info['characteristics']:
                    st.markdown(f"**{char}**")
            
            with col2:
                st.markdown("### 📊 Distribution des Probabilités")
                
                fig = go.Figure(data=[
                    go.Bar(
                        x=list(VARIETY_NAMES.values()),
                        y=probas * 100,
                        marker_color=[VARIETY_INFO[v]['color'] for v in VARIETY_NAMES.values()],
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
            
            # Comparaison des 3 variétés
            st.markdown("### 📊 Comparaison des Variétés")
            
            comparison_df = pd.DataFrame({
                'Variété': ['CO_43', 'Ponmani', 'Delux Ponni'],
                'Probabilité (%)': [f"{p:.1f}%" for p in probas * 100],
                'Rendement Moyen': ['3500-4000', '4000-4500', '4200-4800'],
                'Cycle (jours)': ['130-135', '145-150', '135-140'],
                'Sol Préféré': ['Alluvial', 'Argileux', 'Polyvalent']
            })
            
            st.dataframe(comparison_df, use_container_width=True, hide_index=True)
            
            # Recommandations complémentaires
            st.markdown("### 💡 Conseils Personnalisés")
            
            recommendations = []
            
            if soil_type == "clay" and recommended_variety == "Ponmani":
                recommendations.append("✅ Excellent choix ! Ponmani prospère dans les sols argileux")
            elif soil_type == "alluvial" and recommended_variety == "CO_43":
                recommendations.append("✅ Parfait ! CO_43 est idéal pour les sols alluviaux")
            
            if nursery_type == "wet" and recommended_variety == "Ponmani":
                recommendations.append("✅ La pépinière humide favorisera le développement de Ponmani")
            
            if dap_20days > 50 and recommended_variety == "Delux Ponni":
                recommendations.append("✅ Delux Ponni répondra bien aux apports élevés d'engrais")
            
            if recommendations:
                for rec in recommendations:
                    st.success(rec)
            else:
                st.info("💡 Suivez les bonnes pratiques culturales pour optimiser votre rendement")
            
            # Alternatives
            st.markdown("### 🔄 Variétés Alternatives")
            
            sorted_idx = np.argsort(probas)[::-1]
            
            for idx in sorted_idx[1:3]:
                variety_name = VARIETY_NAMES[idx]
                prob = probas[idx] * 100
                info = VARIETY_INFO[variety_name]
                
                st.markdown(f"""
                <div style="background: #F5F5F5; padding: 1rem; border-radius: 10px; margin: 0.5rem 0; 
                            border-left: 4px solid {info['color']};">
                    <h4 style="margin: 0;">{info['emoji']} {variety_name} ({prob:.1f}%)</h4>
                    <p style="margin: 0.5rem 0; color: #666;">{info['description']}</p>
                </div>
                """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"❌ Erreur : {str(e)}")
            import traceback
            with st.expander("Détails de l'erreur"):
                st.code(traceback.format_exc())
    else:
        st.error("Modèle non disponible")

# Informations complémentaires
with st.expander("ℹ️ À propos des variétés"):
    st.markdown("""
    ### 🌾 CO_43
    - **Origine** : Coimbatore (Inde)
    - **Durée** : 130-135 jours
    - **Rendement** : 3500-4000 kg/ha
    - **Résistance** : Excellente résistance à la sécheresse
    - **Sol idéal** : Alluvial, bien drainé
    
    ### 🌿 Ponmani
    - **Origine** : Tamil Nadu (Inde)
    - **Durée** : 145-150 jours
    - **Rendement** : 4000-4500 kg/ha
    - **Qualité** : Grain premium, très recherché
    - **Sol idéal** : Argileux, riche en eau
    
    ### ⭐ Delux Ponni
    - **Origine** : Hybride amélioré
    - **Durée** : 135-140 jours
    - **Rendement** : 4200-4800 kg/ha
    - **Avantage** : Polyvalent, haut rendement
    - **Sol idéal** : Tous types de sol
    """)

with st.expander("📈 Performance des modèles"):
    st.write("""
    **XGBoost** (Recommandé)
    - Précision : 88.7%
    - Meilleure capacité à capturer les interactions complexes
    - Robuste aux données manquantes
    
    **Random Forest**
    - Précision : 80.5%
    - Bon compromis précision/interprétabilité
    - Moins sensible au surapprentissage
    
    Les modèles ont été validés sur 20% du dataset (split test).
    """)