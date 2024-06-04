#
# BUILDER 
##############################################################################################################
# Ubuntu:22.04
# Use the latest Ubuntu 22.04 version in Docker Hub.
# https://hub.docker.com/_/ubuntu/tags?page=1&name=22.04
ARG UBUNTU_VERSION=83f0c2a8d6f266d687d55b5cb1cb2201148eb7ac449e4202d9646b9083f1cee0

FROM ubuntu@sha256:${UBUNTU_VERSION} as final

RUN apt -y update && \
apt install -y \
python3.10 \
python3.10-venv \
python3-pip

RUN update-alternatives --install \
    /usr/bin/python3 python3 /usr/bin/python3.10 1

ENV PATH=/venv/bin:${PATH}
ENV PYTHONUNBUFFERED=1
COPY requirements.txt /venv/requirements.txt
RUN python3 -m venv /venv && \
pip install --no-cache-dir -r /venv/requirements.txt

COPY publisher.py /

# Use "/sax/test" as default sax cell.
ENV SAX_CELL=/sax/cell

RUN echo '#!/bin/bash\n\
python -m publisher \n' \ 
> /usr/bin/publisher_entrypoint.sh && chmod +x /usr/bin/publisher_entrypoint.sh

ENTRYPOINT ["/usr/bin/publisher_entrypoint.sh"]