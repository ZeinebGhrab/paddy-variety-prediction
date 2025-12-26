@echo off
REM Script pour Windows
REM Usage: run_all.bat [option]

echo ============================================
echo   Systeme d'Aide a la Decision Agricole
echo ============================================
echo.

REM Créer les dossiers
echo Creation des dossiers...
if not exist "data" mkdir data
if not exist "models\regression" mkdir models\regression
if not exist "models\classification" mkdir models\classification
if not exist "models\scalers" mkdir models\scalers
if not exist "output" mkdir output
echo Dossiers crees
echo.

REM Vérifier l'argument
if "%1"=="data" goto DATA
if "%1"=="models" goto MODELS
if "%1"=="app" goto APP
if "%1"=="" goto ALL
goto USAGE

:DATA
echo ============================================
echo   ETAPE 1: PREPARATION DES DONNEES
echo ============================================
echo.

if exist "data\paddydataset.csv" (
    echo 1. Generation des donnees bruitees...
    python src\01_data_generation.py
    echo.
    
    echo 2. Analyse exploratoire...
    python src\02_eda.py
    echo.
    
    echo 3. Nettoyage des donnees...
    python src\03_data_cleaning.py
    echo.
    
    echo Preparation des donnees terminee
) else (
    echo ERREUR: data\paddydataset.csv non trouve
    echo Placez votre fichier de donnees dans le dossier data\
    exit /b 1
)
goto END

:MODELS
echo ============================================
echo   ETAPE 2: ENTRAINEMENT DES MODELES
echo ============================================
echo.

if exist "data\cleaned_paddydataset.csv" (
    echo 1. Entrainement des modeles de regression...
    python src\05_regression_modeling.py
    echo.
    
    echo 2. Entrainement des modeles de classification...
    python src\06_classification_modeling.py
    echo.
    
    echo Entrainement des modeles termine
) else (
    echo ERREUR: data\cleaned_paddydataset.csv non trouve
    echo Executez d'abord la preparation des donnees
    exit /b 1
)
goto END

:APP
echo ============================================
echo   ETAPE 3: LANCEMENT DE L'APPLICATION
echo ============================================
echo.

if exist "models\regression" (
    echo Lancement de Streamlit...
    echo L'application sera accessible a: http://localhost:8501
    echo.
    streamlit run app\streamlit_app.py
) else (
    echo ERREUR: Modeles non trouves
    echo Entrainez d'abord les modeles
    exit /b 1
)
goto END

:ALL
call :DATA
if errorlevel 1 exit /b 1
call :MODELS
if errorlevel 1 exit /b 1
call :APP
goto END

:USAGE
echo Usage: %0 [data^|models^|app]
echo.
echo Options:
echo   data    - Preparation des donnees uniquement
echo   models  - Entrainement des modeles uniquement
echo   app     - Lancement de l'application uniquement
echo   [vide]  - Pipeline complet
exit /b 1

:END
echo.
echo ============================================
echo   TERMINE
echo ============================================
pause