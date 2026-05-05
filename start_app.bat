@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo 포켓몬 순수 Python GUI 분석기를 실행합니다.
echo TensorFlow를 사용하지 않는 버전입니다.
echo.

python pokemon_gui.py

if errorlevel 1 (
    echo.
    echo python 명령어가 실패했습니다. py 명령어로 다시 시도합니다.
    py pokemon_gui.py
)

pause
