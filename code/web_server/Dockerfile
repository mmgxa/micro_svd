FROM python:3.9

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./web_server.py /code/

CMD ["uvicorn", "web_server:app", "--host", "0.0.0.0", "--port", "80"]