FROM python:3.12-slim

WORKDIR /app

# CPU-only torch (much smaller image: ~600MB vs 2GB+ GPU)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements_serve.txt .
RUN pip install --no-cache-dir -r requirements_serve.txt

COPY src/ ./src/

ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "src.mcp.server:app", "--host", "0.0.0.0", "--port", "8765"]
