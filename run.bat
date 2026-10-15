@echo off
rem Runs the whole pipeline after COLMAP (see README for the COLMAP steps).
rem Edit PY if your Python environment is somewhere else.
set PY=C:\v\Scripts\python
cd /d %~dp0
rem Data (videos, images, ws, clouds, output) lives in this same folder unless ROOM_ROOT is set
if not defined ROOM_ROOT set ROOM_ROOT=%~dp0

%PY% detect.py || goto :err
%PY% label.py  || goto :err
if not exist output\scale.json %PY% scale.py
%PY% report.py || goto :err
%PY% viz.py    || goto :err
echo.
echo Done. Results are in the output folder (report.txt, panels, changes_3d.html).
goto :eof

:err
echo.
echo A step failed - see the message above.
exit /b 1
