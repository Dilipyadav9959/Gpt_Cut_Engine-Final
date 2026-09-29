FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY editron_cut_engine ./editron_cut_engine
COPY docs ./docs
EXPOSE 8090
CMD ["uvicorn","editron_cut_engine.api:app","--host","0.0.0.0","--port","8090"]
