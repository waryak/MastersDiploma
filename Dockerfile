FROM continuumio/anaconda3:latest
MAINTAINER "Vladislav Ladenkov <waryak2012@mail.ru>"

WORKDIR MasterDiploma

COPY source ./source
COPY config.yml ./

ENV CONFIG config.yml

# apt-get install rabbitmq-server

#CMD ["python3", "source/main.py"]

