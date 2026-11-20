@echo off
setlocal
pushd "%~dp0" || exit /b 1

set "APP_PYTHON=.venv\Scripts\python.exe"
if not exist "%APP_PYTHON%" (
    py -3.11 -m venv .venv
    if errorlevel 1 goto setup_failed
)

"%APP_PYTHON%" -c "import streamlit, plotly, PIL, numpy, pandas, sklearn, graphviz" >nul 2>&1
if errorlevel 1 (
    "%APP_PYTHON%" -m pip install --disable-pip-version-check -r requirements.txt
    if errorlevel 1 goto setup_failed
)

"%APP_PYTHON%" -m streamlit run app.py %*
set "APP_EXIT=%errorlevel%"
popd
exit /b %APP_EXIT%

:setup_failed
echo CottonLeaf AI setup failed. Install Python 3.11 with the Windows py launcher, then run this file from a terminal.
popd
exit /b 1
