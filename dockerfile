FROM quay.io/pennsignals/alpine-3.8-python-3.7-machinelearning-mssql:5.0
WORKDIR /tmp
COPY readme.md .
COPY setup.cfg .
COPY setup.py .
COPY .git ./.git
COPY local ./local
COPY src ./src
COPY tests ./tests
RUN apk add --no-cache --virtual .build \
        git \
    && pip install -U pytest \
    && pip install --no-cache-dir "." \
    && apk del --no-cache .build
