FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Переписки, сессия Telegram и очередь лидов лежат в data/ и монтируются снаружи
VOLUME ["/app/data"]

CMD ["python", "-m", "evrika"]
