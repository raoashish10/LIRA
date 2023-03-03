FROM python:3.10.8
ENV pythonunbuffered 1

WORKDIR /lira-docker
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
RUN python -m nltk.downloader punkt
CMD python manage.py runserver 0.0.0.0:80