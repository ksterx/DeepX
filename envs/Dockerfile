FROM nvcr.io/nvidia/pytorch:23.05-py3

LABEL maintainer="ksterx <kostonerx@gmail.com>"

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV TZ=Asia/Tokyo

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    ffmpeg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set timezone as $TZ
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime \
    && echo ${TZ} > /etc/timezone

# Install dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip \
    && pip install -r /tmp/requirements.txt

RUN echo "set completion-ignore-case on" >> ${HOME}/.inputrc

CMD ["/bin/bash"]
