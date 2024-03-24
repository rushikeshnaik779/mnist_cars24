FROM python:3.11
WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["python", "inference_api.py"]
#CMD ["uvicorn", "api/inference_api:app", "--host", "0.0.0.0", "--port", "8000"]
