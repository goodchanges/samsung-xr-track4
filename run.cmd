@echo off
if exist "%~dp0.venv\Scripts\python.exe" (
  "%~dp0.venv\Scripts\python.exe" "%~dp0code\safe_run.py" %*
) else (
  python "%~dp0code\safe_run.py" %*
)
