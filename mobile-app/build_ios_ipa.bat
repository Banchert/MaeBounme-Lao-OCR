@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ========================================================
echo   Build iOS IPA - Mae Bounmee (Scan) via Expo EAS Cloud
echo   (Requires Apple Developer Account for real iPhone/App Store)
echo ========================================================
call npx eas-cli whoami >nul 2>&1
if errorlevel 1 (
    echo Please log in to your Expo account (https://expo.dev):
    call npx eas-cli login
)
call npx eas-cli build --platform ios --profile production
pause
