import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

# Configuration de la page
st.set_page_config(
    page_title="Système d'Aide à la Décision Agricole",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS personnalisé
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #2E7D32;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #A5D6A7 0%, #66BB6A 100%);
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .info-box {
        background-color: #E8F5E9;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #4CAF50;
        margin: 1rem 0;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# Titre principal
st.markdown('<h1 class="main-header">🌾 Système d\'Aide à la Décision Agricole pour la Culture du Riz 🌱</h1>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://raw.githubusercontent.com/twitter/twemoji/master/assets/72x72/1f33e.png", width=100)
    st.title("Navigation")
    st.markdown("---")
    
    st.info("""
    ### 📱 À propos
    
    Cette application aide les agriculteurs à :
    - 📊 **Prédire le rendement** de leur culture
    - 🌱 **Choisir la meilleure variété** de riz
    - 📈 **Analyser leurs données** agronomiques
    
    Développé avec ❤️ pour les agriculteurs
    """)
    
    st.markdown("---")
    st.markdown("### 📞 Contact")
    st.markdown("🌾 Support Agricole")
    st.markdown("📧 support@paddy-ai.tn")

# Page d'accueil
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="metric-card">
        <h2>🎯</h2>
        <h3>Prédiction Précise</h3>
        <p>Modèles ML entraînés sur des milliers de parcelles</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card">
        <h2>🌾</h2>
        <h3>3 Variétés</h3>
        <p>CO_43, Ponmani, Delux Ponni</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card">
        <h2>📊</h2>
        <h3>Analyse Complète</h3>
        <p>Facteurs météo, sol, intrants</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Section informations
st.markdown("""
<div class="info-box">
    <h2>🚀 Comment utiliser cette application ?</h2>
    
    <h3>1️⃣ Prédiction du Rendement</h3>
    <p>Entrez les caractéristiques de votre parcelle (météo, sol, intrants) pour obtenir une estimation du rendement en kg.</p>
    
    <h3>2️⃣ Recommandation de Variété</h3>
    <p>Découvrez quelle variété de riz (CO_43, Ponmani ou Delux Ponni) est la mieux adaptée à vos conditions.</p>
    
    <h3>3️⃣ Analyse des Données</h3>
    <p>Visualisez et analysez vos données historiques pour mieux comprendre les facteurs de réussite.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# Section caractéristiques
st.header("✨ Fonctionnalités Principales")

col1, col2 = st.columns(2)

with col1:
    st.subheader("🔮 Intelligence Artificielle")
    st.write("""
    - **XGBoost** pour la classification (ROC-AUC: 0.87)
    - **Ridge Regression** pour les prédictions de rendement (R²: 0.89)
    - Modèles entraînés sur données réelles
    - Validation croisée rigoureuse
    """)
    
    st.subheader("🌡️ Facteurs Météorologiques")
    st.write("""
    - Précipitations par période
    - Températures min/max
    - Humidité et vent
    - Direction du vent
    """)

with col2:
    st.subheader("🌱 Facteurs Agronomiques")
    st.write("""
    - Type de sol (alluvial, argileux)
    - Superficie de la parcelle
    - Méthode de pépinière
    - Bloc agricole
    """)
    
    st.subheader("💊 Intrants")
    st.write("""
    - Engrais (DAP, Urée, Potasse)
    - Micronutriments
    - Pesticides et herbicides
    - Taux de semis
    """)

st.markdown("---")

# Instructions
st.header("📖 Guide d'Utilisation")

with st.expander("📝 Préparer vos données"):
    st.write("""
    Avant d'utiliser l'application, assurez-vous d'avoir les informations suivantes :
    
    **Données météorologiques** (sur la période de culture) :
    - Précipitations totales
    - Températures minimales et maximales
    - Humidité moyenne
    - Vitesse du vent
    
    **Informations sur la parcelle** :
    - Superficie en hectares
    - Type de sol
    - Bloc agricole
    - Méthode de pépinière (sèche/humide)
    
    **Intrants appliqués** :
    - Quantités d'engrais (DAP, Urée, Potasse)
    - Micronutriments
    - Pesticides et herbicides
    """)

with st.expander("🎯 Interpréter les résultats"):
    st.write("""
    **Prédiction de Rendement** :
    - Un rendement élevé (> 4000 kg) indique de bonnes conditions
    - Un rendement moyen (2500-4000 kg) est acceptable
    - Un rendement faible (< 2500 kg) nécessite des ajustements
    
    **Recommandation de Variété** :
    - **CO_43** : Adapté aux sols alluviaux, résistant à la sécheresse
    - **Ponmani** : Préfère les sols argileux humides
    - **Delux Ponni** : Polyvalent, rendement élevé
    
    L'application affiche également un score de confiance pour chaque recommandation.
    """)

st.markdown("---")

# Footer
st.markdown("""
<div style="text-align: center; padding: 2rem; background-color: #F1F8E9; border-radius: 10px;">
    <p style="font-size: 1.2rem; color: #558B2F;">
        🌾 <strong>Cultivons l'avenir ensemble</strong> 🌱
    </p>
    <p style="color: #7CB342;">
        Version 1.0 | Développé pour les agriculteurs tunisiens
    </p>
</div>
""", unsafe_allow_html=True)

# Instructions pour commencer
st.sidebar.markdown("---")
st.sidebar.success("👈 Sélectionnez une page pour commencer !")