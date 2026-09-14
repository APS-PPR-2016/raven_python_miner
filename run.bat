
REM --- Setup Python Virtual Environment ---
set VENV_DIR=.venv

    echo  - Activating virtual environment-run batch jobs again, if it fails...
    call "%VENV_DIR%\Scripts\activate.bat"
    powershell -NoExit -Command "& { Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process; . \"%VENV_DIR%\Scripts\Activate.ps1\" }"
    echo  - Installing/verifying dependencies...
    pip install -r requirements.txt
    echo  - Installing project in editable mode...
    pip install -e .
    echo.
    if %errorlevel% neq 0 (
        echo !RED![!] Failed to activate the virtual environment. Halting execution.!RESET!
        exit /b
    )