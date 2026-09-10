@echo off
python -m venv .venv
call .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
echo.
echo Environment ready.
echo Run one of these:
echo   python pf8_pipeline.py
echo   python pf8_pipeline.py --quick
echo   python pf8_pipeline.py --skip-download --skip-filter
pause
