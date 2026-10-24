@echo off
echo ========================================================
echo Running Automated Reliability Monitor Test Suite
echo ========================================================
python -m pytest -v tests
pause
