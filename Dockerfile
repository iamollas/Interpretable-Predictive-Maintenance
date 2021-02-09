FROM ubuntu:18.04
FROM python:3.7.8
ARG NB_USER=jovyan
ARG NB_UID=1000
ENV USER ${NB_USER}
ENV NB_UID ${NB_UID}
ENV HOME /home/${NB_USER}

RUN adduser --disabled-password \
    --gecos "Default user" \
    --uid ${NB_UID} \
    ${NB_USER}
    
    
RUN apt-get update
RUN apt-get dist-upgrade -y

RUN DEBIAN_FRONTEND=noninteractive apt-get -y dist-upgrade
RUN DEBIAN_FRONTEND=noninteractive apt-get -y install software-properties-common
RUN DEBIAN_FRONTEND=noninteractive apt-add-repository ppa:swi-prolog/stable
RUN DEBIAN_FRONTEND=noninteractive apt-get -y install swi-prolog

WORKDIR /usr/src/app

COPY TEDS_RUL .

RUN mkdir prolog_outputs

RUN pip install --upgrade setuptools

RUN pip install --no-cache-dir -r requirements.txt

ADD TEDS_RUL/lime_tabular.py /usr/local/lib/python3.7/site-packages/lime

RUN python3 -m pip install --upgrade notebook

# Make sure the contents of our repo are in ${HOME}
COPY . ${HOME}
USER root
RUN chown -R ${NB_UID} ${HOME}
USER ${NB_USER}
