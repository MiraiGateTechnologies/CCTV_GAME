@echo off
echo ========================================
echo   LiveKit Server — CCTV Casino Game
echo   URL: ws://localhost:7880
echo   Key: devkey / Secret: cctv-game-secret-key-32chars-min!
echo ========================================
echo.
cd /d "%~dp0"
livekit_server\livekit-server.exe --config livekit.yaml
pause
