@echo off
setlocal enabledelayedexpansion

set "SCRIPT_DIR=%~dp0"
set "PYTHON_BIN=%PYTHON%"
if "%PYTHON_BIN%"=="" set "PYTHON_BIN=python"

where %PYTHON_BIN% >nul 2>&1
if errorlevel 1 (
  echo [quick-native][error] Python not found: %PYTHON_BIN%
  echo [quick-native][hint] Install Python 3.10+ or set PYTHON env var ^(e.g. set PYTHON=py -3^).
  exit /b 1
)

cd /d "%SCRIPT_DIR%"
%PYTHON_BIN% quick_native.py %*
exit /b %errorlevel%
