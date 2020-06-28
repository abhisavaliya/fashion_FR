FROM ubuntu:latest
MAINTAINER Abhi_Savaliya <abhisavaliya01@gmail.com>

FROM python:3.6

WORKDIR /fashion_fr
COPY Fashion_Similarity/ .
COPY requirements.txt .

RUN pip install --upgrade pip && apt-get update && apt-get -y update && pip install -r requirements.txt && apt-get clean all

CMD ["jupyter", "notebook", "--port=8888", "--ip=0.0.0.0", "--allow-root"]
