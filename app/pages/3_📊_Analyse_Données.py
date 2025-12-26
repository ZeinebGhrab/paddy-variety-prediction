import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(page_title="Analyse Données", page_icon="📊", layout="wide")

# Titre
st.title("📊 Analyse des Données Agricoles")
st.markdown("---")

# Charger les données
@st.cache_data
def load_data():
    """Charge les données nettoyées"""
    try:
        df = pd.read_csv("data/cleaned_paddydataset.csv")
        return df
    except Exception as e:
        st.error(f"Erreur lors du chargement des données : {e}")
        return None

df = load_data()

if df is not None:
    # Sidebar - Filtres
    with st.sidebar:
        st.header("🔍 Filtres")
        
        # Filtre par variété (si la colonne existe)
        variety_cols = [col for col in df.columns if 'Variety' in col or 'variety' in col.lower()]
        if variety_cols:
            varieties = st.multiselect(
                "Variétés",
                options=['Toutes'] + list(df[variety_cols[0]].unique()) if variety_cols else ['Toutes'],
                default=['Toutes']
            )
        
        # Filtre par rendement
        if 'Paddy yield(in Kg)' in df.columns:
            min_yield, max_yield = st.slider(
                "Plage de rendement (kg)",
                int(df['Paddy yield(in Kg)'].min()),
                int(df['Paddy yield(in Kg)'].max()),
                (int(df['Paddy yield(in Kg)'].min()), int(df['Paddy yield(in Kg)'].max()))
            )
            
            # Filtrer le dataframe
            df_filtered = df[
                (df['Paddy yield(in Kg)'] >= min_yield) & 
                (df['Paddy yield(in Kg)'] <= max_yield)
            ]
        else:
            df_filtered = df.copy()
        
        st.markdown("---")
        st.info(f"📊 **{len(df_filtered)}** parcelles sélectionnées")
    
    # Onglets
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Vue d'Ensemble", "🔗 Corrélations", "📊 Distributions", "📥 Export"])
    
    # TAB 1 : Vue d'ensemble
    with tab1:
        st.header("📈 Statistiques Générales")
        
        # Métriques principales
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "📊 Total Parcelles",
                f"{len(df_filtered):,}",
                f"{len(df_filtered) - len(df):+,}"
            )
        
        with col2:
            if 'Paddy yield(in Kg)' in df_filtered.columns:
                avg_yield = df_filtered['Paddy yield(in Kg)'].mean()
                st.metric(
                    "🌾 Rendement Moyen",
                    f"{avg_yield:.0f} kg",
                    f"{avg_yield - df['Paddy yield(in Kg)'].mean():+.0f} kg"
                )
        
        with col3:
            if 'Hectares ' in df_filtered.columns or 'Hectares' in df_filtered.columns:
                hectares_col = 'Hectares ' if 'Hectares ' in df_filtered.columns else 'Hectares'
                total_area = df_filtered[hectares_col].sum()
                st.metric(
                    "📏 Surface Totale",
                    f"{total_area:.1f} ha"
                )
        
        with col4:
            numeric_cols = df_filtered.select_dtypes(include=[np.number]).columns
            st.metric(
                "🔢 Variables",
                len(df_filtered.columns),
                f"{len(numeric_cols)} numériques"
            )
        
        st.markdown("---")
        
        # Graphiques de vue d'ensemble
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Distribution du Rendement")
            if 'Paddy yield(in Kg)' in df_filtered.columns:
                fig = px.histogram(
                    df_filtered,
                    x='Paddy yield(in Kg)',
                    nbins=30,
                    title="Histogramme du Rendement",
                    labels={'Paddy yield(in Kg)': 'Rendement (kg)'}
                )
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("📈 Box Plot Rendement")
            if 'Paddy yield(in Kg)' in df_filtered.columns:
                fig = px.box(
                    df_filtered,
                    y='Paddy yield(in Kg)',
                    title="Dispersion du Rendement",
                    labels={'Paddy yield(in Kg)': 'Rendement (kg)'}
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Tableau de statistiques descriptives
        st.subheader("📋 Statistiques Descriptives")
        numeric_cols = df_filtered.select_dtypes(include=[np.number]).columns
        st.dataframe(
            df_filtered[numeric_cols].describe().round(2),
            use_container_width=True
        )
    
    # TAB 2 : Corrélations
    with tab2:
        st.header("🔗 Analyse des Corrélations")
        
        numeric_cols = df_filtered.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) > 1:
            # Sélection des variables
            col1, col2 = st.columns(2)
            
            with col1:
                var_x = st.selectbox(
                    "Variable X",
                    numeric_cols,
                    index=0
                )
            
            with col2:
                var_y = st.selectbox(
                    "Variable Y",
                    numeric_cols,
                    index=min(1, len(numeric_cols)-1)
                )
            
            # Scatter plot
            if var_x and var_y:
                fig = px.scatter(
                    df_filtered,
                    x=var_x,
                    y=var_y,
                    title=f"Relation entre {var_x} et {var_y}",
                    trendline="ols",
                    labels={var_x: var_x, var_y: var_y}
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Calcul corrélation
                corr = df_filtered[[var_x, var_y]].corr().iloc[0, 1]
                
                if abs(corr) > 0.7:
                    st.success(f"✅ Forte corrélation : {corr:.3f}")
                elif abs(corr) > 0.4:
                    st.info(f"ℹ️ Corrélation modérée : {corr:.3f}")
                else:
                    st.warning(f"⚠️ Faible corrélation : {corr:.3f}")
            
            st.markdown("---")
            
            # Matrice de corrélation
            st.subheader("🔥 Matrice de Corrélation")
            
            # Sélection de variables pour la matrice
            selected_vars = st.multiselect(
                "Sélectionnez les variables (max 10)",
                numeric_cols,
                default=numeric_cols[:min(10, len(numeric_cols))]
            )
            
            if selected_vars and len(selected_vars) > 1:
                corr_matrix = df_filtered[selected_vars].corr()
                
                fig = px.imshow(
                    corr_matrix,
                    text_auto='.2f',
                    aspect="auto",
                    color_continuous_scale='RdBu_r',
                    title="Matrice de Corrélation"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Top corrélations
                st.subheader("📊 Top Corrélations")
                
                # Extraire les corrélations
                corr_pairs = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        corr_pairs.append({
                            'Variable 1': corr_matrix.columns[i],
                            'Variable 2': corr_matrix.columns[j],
                            'Corrélation': corr_matrix.iloc[i, j]
                        })
                
                corr_df = pd.DataFrame(corr_pairs)
                corr_df = corr_df.sort_values('Corrélation', key=abs, ascending=False)
                
                st.dataframe(
                    corr_df.head(10),
                    use_container_width=True,
                    hide_index=True
                )
        else:
            st.warning("Pas assez de variables numériques pour l'analyse de corrélation")
    
    # TAB 3 : Distributions
    with tab3:
        st.header("📊 Distributions des Variables")
        
        numeric_cols = df_filtered.select_dtypes(include=[np.number]).columns.tolist()
        
        if numeric_cols:
            selected_var = st.selectbox(
                "Sélectionnez une variable",
                numeric_cols
            )
            
            if selected_var:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Histogramme
                    fig = px.histogram(
                        df_filtered,
                        x=selected_var,
                        nbins=30,
                        title=f"Distribution de {selected_var}",
                        labels={selected_var: selected_var}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Box plot
                    fig = px.box(
                        df_filtered,
                        y=selected_var,
                        title=f"Box Plot de {selected_var}",
                        labels={selected_var: selected_var}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Statistiques
                st.subheader("📈 Statistiques")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Moyenne", f"{df_filtered[selected_var].mean():.2f}")
                
                with col2:
                    st.metric("Médiane", f"{df_filtered[selected_var].median():.2f}")
                
                with col3:
                    st.metric("Écart-type", f"{df_filtered[selected_var].std():.2f}")
                
                with col4:
                    st.metric("IQR", f"{df_filtered[selected_var].quantile(0.75) - df_filtered[selected_var].quantile(0.25):.2f}")
        else:
            st.warning("Aucune variable numérique disponible")
    
    # TAB 4 : Export
    with tab4:
        st.header("📥 Export des Données")
        
        st.info("""
        💡 **Conseil** : Vous pouvez exporter les données filtrées pour une analyse externe.
        """)
        
        # Options d'export
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Données Brutes")
            
            # Bouton téléchargement CSV
            csv = df_filtered.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Télécharger CSV",
                data=csv,
                file_name="donnees_filtrees.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            st.subheader("📈 Statistiques")
            
            # Statistiques descriptives
            stats = df_filtered.describe()
            stats_csv = stats.to_csv().encode('utf-8')
            
            st.download_button(
                label="📥 Télécharger Statistiques",
                data=stats_csv,
                file_name="statistiques.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        st.markdown("---")
        
        # Aperçu des données
        st.subheader("👁️ Aperçu des Données")
        
        n_rows = st.slider("Nombre de lignes à afficher", 5, 50, 10)
        st.dataframe(df_filtered.head(n_rows), use_container_width=True)
        
        st.caption(f"Affichage de {n_rows} lignes sur {len(df_filtered)} au total")

else:
    st.error("❌ Impossible de charger les données")
    st.info("""
    **Vérifiez que :**
    - Le fichier `data/cleaned_paddydataset.csv` existe
    - Le nettoyage des données a été effectué
    - Vous avez les bonnes permissions
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>💡 <strong>Astuce</strong> : Utilisez les filtres dans la barre latérale pour affiner votre analyse</p>
</div>
""", unsafe_allow_html=True)