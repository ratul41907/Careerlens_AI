@echo off
echo ========================================
echo Installing CareerLens AI Dependencies
echo ========================================

cd E:\careerlens-ai

echo.
echo Activating virtual environment...
call venv\Scripts\activate

echo.
echo Upgrading pip...
python -m pip install --upgrade pip

echo.
echo Installing core dependencies...
pip install streamlit pandas numpy requests

echo.
echo Installing PDF/Document processing...
pip install pypdf pypdf2 python-docx openpyxl

echo.
echo Installing NLP and ML...
pip install sentence-transformers scikit-learn nltk

echo.
echo Installing API frameworks...
pip install fastapi uvicorn python-multipart

echo.
echo Installing utilities...
pip install cachetools python-dotenv pydantic

echo.
echo Installing testing tools...
pip install pytest pytest-cov pytest-mock pytest-asyncio coverage faker psutil

echo.
echo ========================================
echo Installation Complete!
echo ========================================
echo.
echo Verifying installations...
pip list | findstr "streamlit pypdf sentence-transformers pytest"

echo.
echo Ready to run! Start with:
echo   streamlit run streamlit_app\Home.py
echo.

pause