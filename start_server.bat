@echo off
echo Dang khoi dong server AgriTech AI...
docker run -p 8000:8000 --env-file .env agritech_ai
pause