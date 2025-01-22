FROM python:3.10

COPY requirements.txt requirements.txt
COPY test.py test.py
COPY docker_run.sh docker_run.sh
RUN pip install -r requirements.txt

RUN chmod +x /docker_run.sh

ENTRYPOINT ["/docker_run.sh"]