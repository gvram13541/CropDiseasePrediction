@echo off

:: Start the Python app in the backend folder
start cmd /k "cd backend && python app.py"

:: Start the npm server in the frontend folder
start cmd /k "cd frontend && npm start"

:: Keep the window open
pause