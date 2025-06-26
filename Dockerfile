FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .

# Install with PyTorch CPU wheels enabled
RUN pip install --no-cache-dir -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
