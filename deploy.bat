@echo off
echo.
echo ================================================
echo   Football Predictor - Deploy automatico
echo ================================================
echo.

cd /d "C:\Users\Francisco Barrera\Downloads\football_predictor"

echo [1/4] Sincronizando con GitHub...
git pull origin main --no-rebase

echo [2/4] Añadiendo cambios...
git add .

echo [3/4] Creando commit...
git commit -m "Update %date% %time%"

echo [4/4] Subiendo a GitHub...
git push

echo.
echo ================================================
echo   Listo! Streamlit redeploya en ~1 minuto
echo   URL: https://ticfbminf-spec-football-p-downloadsfootball-predictorapp-xvmsmj.streamlit.app
echo ================================================
echo.
pause
