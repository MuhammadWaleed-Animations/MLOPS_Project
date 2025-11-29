@echo off
REM Setup script for Building Classifier (Windows)

echo Setting up Building Classifier...

REM Create virtual environment
echo Creating virtual environment...
python -m venv venv

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Install dependencies
echo Installing dependencies...
python -m pip install --upgrade pip
pip install -r requirements.txt

REM Create necessary directories
echo Creating directories...
if not exist "data" mkdir data
if not exist "artifacts" mkdir artifacts
if not exist "uploads" mkdir uploads
if not exist "templates" mkdir templates

echo Setup complete!
echo.
echo Next steps:
echo 1. Add your training images to the 'data' folder
echo 2. Run 'python train.py' to train the model
echo 3. Run 'python app.py' to start the web application
echo 4. Open http://localhost:8000 in your browser

pause

