FROM python:3.12.2 AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1
WORKDIR /app


RUN python -m venv .venv
COPY requirements.txt ./
RUN .venv/bin/pip install -r requirements.txt

FROM python:3.12.2-slim
WORKDIR /app

# Install OpenCV dependencies
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y

COPY --from=builder /app/.venv .venv/
COPY . .
CMD ["/app/.venv/bin/streamlit", "run", "app.py"]
