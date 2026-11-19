@echo off
cd /d "%~dp0"

:loop
echo [%DATE% %TIME%] Запуск бота...
python bot.py

echo [%DATE% %TIME%] Бот упал или завершил работу. Перезапуск через 5 секунд...
timeout /t 5 /nobreak >nul
goto loop
