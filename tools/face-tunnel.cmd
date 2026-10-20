@echo off
chcp 65001 >nul
title Face Recognition - SSH Tunnel
echo.
echo   Tunnel:  http://localhost:8080  ==^>  VM port 80
echo   Keep this window OPEN while using the page.
echo   Close it to close the tunnel.
echo.
echo   Password will be asked below.
echo.
ssh -N -L 8080:127.0.0.1:80 -p 2222 chen@127.0.0.1
echo.
echo Tunnel closed.
pause
