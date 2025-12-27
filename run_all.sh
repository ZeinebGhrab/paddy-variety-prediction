
#!/bin/bash

# Script pour exécuter tout le pipeline
# Usage: bash run_all.sh [option]
# Options:
#   data    - Préparation des données uniquement
#   models  - Entraînement des modèles uniquement
#   app     - Lancement de l'application uniquement
#   all     - Tout (par défaut)

echo "============================================"
echo "  Système d'Aide à la Décision Agricole"
echo "============================================"
echo ""

# Fonction pour créer les dossiers nécessaires
create_dirs() {
    echo "📁 Création des dossiers..."
    mkdir -p data
    mkdir -p models/regression
    mkdir -p models/classification
    mkdir -p models/scalers
    mkdir -p output
    echo "✓ Dossiers créés"
    echo ""
}

# Fonction pour préparer les données
prepare_data() {
    echo "============================================"
    echo "  ÉTAPE 1: PRÉPARATION DES DONNÉES"
    echo "============================================"
    echo ""
    
    if [ -f "data/paddydataset.csv" ]; then
        echo "1️⃣ Génération des données bruitées..."
        python src/01_data_generation.py
        echo ""
        
        echo "2️⃣ Analyse exploratoire..."
        python src/02_eda.py
        echo ""
        
        echo "3️⃣ Nettoyage des données..."
        python src/03_data_cleaning.py
        echo ""
        
        echo "✓ Préparation des données terminée"
    else
        echo "❌ Erreur: data/paddydataset.csv non trouvé"
        echo "   Placez votre fichier de données dans le dossier data/"
        exit 1
    fi
}

# Fonction pour entraîner les modèles
train_models() {
    echo ""
    echo "============================================"
    echo "  ÉTAPE 2: ENTRAÎNEMENT DES MODÈLES"
    echo "============================================"
    echo ""
    
    if [ -f "data/cleaned_paddydataset.csv" ]; then
        echo "1️⃣ Entraînement des modèles de régression..."
        python src/05_regression_modeling.py
        echo ""
        
        echo "2️⃣ Entraînement des modèles de classification..."
        python src/06_classification_modeling.py
        echo ""
        
        echo "✓ Entraînement des modèles terminé"
    else
        echo "❌ Erreur: data/cleaned_paddydataset.csv non trouvé"
        echo "   Exécutez d'abord la préparation des données"
        exit 1
    fi
}

# Fonction pour lancer l'application
launch_app() {
    echo ""
    echo "============================================"
    echo "  ÉTAPE 3: LANCEMENT DE L'APPLICATION"
    echo "============================================"
    echo ""
    
    if [ -d "models/regression" ] && [ -d "models/classification" ]; then
        echo "🚀 Lancement de Streamlit..."
        echo "   L'application sera accessible à: http://localhost:8501"
        echo ""
        streamlit run app/streamlit_app.py
    else
        echo "❌ Erreur: Modèles non trouvés"
        echo "   Entraînez d'abord les modèles"
        exit 1
    fi
}

# Traitement des arguments
case "${1:-all}" in
    data)
        create_dirs
        prepare_data
        ;;
    models)
        create_dirs
        train_models
        ;;
    app)
        launch_app
        ;;
    all)
        create_dirs
        prepare_data
        train_models
        launch_app
        ;;
    *)
        echo "Usage: $0 {data|models|app|all}"
        echo ""
        echo "Options:"
        echo "  data    - Préparation des données uniquement"
        echo "  models  - Entraînement des modèles uniquement"
        echo "  app     - Lancement de l'application uniquement"
        echo "  all     - Pipeline complet (par défaut)"
        exit 1
        ;;
esac

echo ""
echo "============================================"
echo "  ✓ TERMINÉ"
echo "============================================"