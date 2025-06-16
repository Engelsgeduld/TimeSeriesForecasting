FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
COPY . /app
RUN pip install --no-cache-dir -r requirements.txt
ENTRYPOINT ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]