FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

ENV NB_USER jovyan
ENV NB_UID 1000
ENV NB_PREFIX /

RUN export DEBIAN_FRONTEND=noninteractive \
    && apt-get -yq update \
    && apt-get -yq install --no-install-recommends \
        git \
        ffmpeg \
        software-properties-common \
    &&  apt upgrade -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN export DEBIAN_FRONTEND=noninteractive \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get -yq update \
    && apt-get -yq install --no-install-recommends \
        python3.10 \
        python3.10-venv \
    &&  apt upgrade -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -M -s /bin/bash -N -u ${NB_UID} ${NB_USER} \
    && mkdir -p ${HOME} \
    && chown -R ${NB_USER}:users ${HOME} \
    && chown -R ${NB_USER}:users /usr/local/bin

USER $NB_UID
WORKDIR /home/${NB_USER}/
ENV PATH=/venv/bin:$PATH

COPY --chown=${NB_USER}:users ./requirements.txt /home/${NB_USER}/requirements.txt
RUN python3.10 -m venv /home/${NB_USER}/venv \
    && /home/${NB_USER}/venv/bin/pip install --upgrade pip wheel \
    && /home/${NB_USER}/venv/bin/pip install -r /home/${NB_USER}/requirements.txt

COPY --chown=${NB_USER}:users ./.env* /home/${NB_USER}/.env
COPY --chown=${NB_USER}:users ./src /home/${NB_USER}/src

EXPOSE 7860

CMD ["/home/jovyan/venv/bin/python", "-u", "/home/jovyan/src/app.py"]