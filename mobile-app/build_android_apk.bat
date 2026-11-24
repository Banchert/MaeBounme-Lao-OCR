@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ========================================================
echo   Build Android APK - Mae Bounmee (Scan) via Expo EAS
echo ========================================================
call npx eas-cli whoami >nul 2>&1
if errorlevel 1 (
    echo Please log in to your free Expo account (https://expo.dev):
    call npx eas-cli login
)
call npx eas-cli build --platform android --profile preview
pause
