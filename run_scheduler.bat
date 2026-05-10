@echo off
:: run_scheduler.bat — Start Growth Chamber CV daily scheduler
:: Called by Windows Task Scheduler on system startup.
:: Activates the virtual environment and runs scheduler_final.py.

cd /d "C:\Users\LukeB\OneDrive\Desktop\Growth_Chamber_cv"

"scripts\.venv\Scripts\python.exe" "scripts\scheduler_final.py" >> "results\scheduler_log.txt" 2>&1
