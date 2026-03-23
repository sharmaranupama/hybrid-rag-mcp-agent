FROM python:3.11-slim
WORKDIR /app
 
RUN apt-get update && apt-get install -y --no-install-recommends \
    nodejs npm build-essential \
    && rm -rf /var/lib/apt/lists/*
 
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
 
COPY . .
 
EXPOSE 6274 6277 8501
ENV HOST=0.0.0.0
ENV PYTHONUNBUFFERED=1
 
CMD ["npx", "-y", "@modelcontextprotocol/inspector", "python3", "/app/mcp_server.py"]