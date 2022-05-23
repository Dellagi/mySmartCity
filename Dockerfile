FROM python:3.9-buster

WORKDIR /code
RUN apt-get clean && apt-get update --allow-releaseinfo-change --allow-insecure-repositories
RUN apt-get install -y apt-utils gcc musl-dev linux-headers-amd64 libc-dev g++ make

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
EXPOSE 8000
EXPOSE 8080
COPY . .
RUN /bin/bash -c 'chmod +x /code/backend/*.py'
RUN /bin/bash -c 'chmod +x /code/entrypoint.sh'

CMD ["/bin/bash", "entrypoint.sh"]
