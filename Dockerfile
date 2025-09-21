FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY main.py .
COPY data/ data/  # Include PDF if small; otherwise mount as volume

CMD ["python", "main.py"]