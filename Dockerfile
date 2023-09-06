FROM python:3.10-slim-buster
LABEL authors="Arturo Ortiz"

WORKDIR /app

COPY requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

RUN python -m pytest -p no:warnings

CMD ["python", "main.py", "-h"]

ENTRYPOINT ["python", "main.py"]