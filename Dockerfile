FROM nvidia/cuda:11.6.0-cudnn8-devel-ubuntu20.04

LABEL maintainer "Christian Behrens <c.behrens@uni-tuebingen.de>"

USER root

# Set the time zone correctly
ENV TZ=Europe/Berlin
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

ENV SHELL /bin/bash
ENV PIP_ROOT_USER_ACTION=ignore

# Install SSH, sudo and gfortran, etc.
RUN apt-get update -qq \
    && DEBIAN_FRONTEND=noninteractive apt-get install -yq -qq --no-install-recommends \
    ca-certificates \
    gfortran \
    openssh-server \
    pwgen \
    screen \
    sudo \
    tmux \
    vim \
    wget \
    xterm \
    git \
    zsh \
    build-essential \
    curl \
    libcurl3-dev \
    libfreetype6-dev \
    libpng-dev \
    libzmq3-dev \
    pkg-config \
    rsync \
    software-properties-common \
    unzip \
    zip \
    zlib1g-dev \
    libjs-mathjax \
    libgoogle-perftools-dev \
    libblas-dev \
    liblapack-dev \
    swig \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*


RUN wget https://github.com/robbyrussell/oh-my-zsh/raw/master/tools/install.sh -O - | zsh || true
RUN echo "export SHELL=/bin/zsh" >> /root/.bashrc
RUN echo "exec /bin/zsh -l" >> /root/.bashrc
RUN echo 'PROMPT="$fg[cyan]%}root@%{$fg[blue]%}%m ${PROMPT}"' >> /root/.zshrc

# Use pyenv to install latest version of Python 
RUN git clone --depth=1 https://github.com/pyenv/pyenv.git .pyenv
ENV PYENV_ROOT="/.pyenv"
ENV PATH="$PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH"
# Changed from berenslab/deeplearning: neuralpredictors/nnfabrik requires iterable abstract base class, which was deprecated in python 3.10
# RUN pyenv install miniconda3-latest
# RUN pyenv global miniconda3-latest
RUN pyenv install miniconda3-3.9-4.12.0
RUN pyenv global miniconda3-3.9-4.12.0


# fix invalid pointer bug in Tensorboard
ENV LD_PRELOAD /usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4

# Enable passwordless sudo for all users
RUN echo '%sudo ALL=(ALL:ALL) NOPASSWD:ALL' >> /etc/sudoers

# Setup gosu (https://github.com/tianon/gosu)
# gosu is an improved version of su which behaves better inside docker
# we use it to dynamically switch to the desired user in the entrypoint
# (see below)
ENV GOSU_VERSION 1.16
# Use unsecure HTTP via Port 80 to fetch key due to firewall in CIN.
RUN set -x \
    && dpkgArch="$(dpkg --print-architecture | awk -F- '{ print $NF }')" \
    && wget -O /usr/local/bin/gosu "https://github.com/tianon/gosu/releases/download/$GOSU_VERSION/gosu-$dpkgArch" \
    && chmod +x /usr/local/bin/gosu \
    && gosu nobody true

COPY dlgn_cnn/docker_scripts/entrypoint.sh /usr/local/bin/
RUN chmod a+x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

RUN mkdir /usr/.jupyter
ENV JUPYTER_CONFIG_DIR /usr/.jupyter
COPY dlgn_cnn/docker_scripts/jupyter_notebook_config.py /usr/.jupyter/
COPY dlgn_cnn/docker_scripts/jupyter_server_config.py /usr/.jupyter/

RUN pip install --upgrade pip
RUN pip --no-cache-dir install \
    tqdm \
    ipykernel \
    jupyter \
    jupyterlab \
    matplotlib \
    numpy \
    scipy \
    scikit-learn \
    pandas \
    seaborn \
    click \
    mypy \
    boltons \
    tensorflow \
    opencv-contrib-python \
    six \
    wheel \
    deeplake \
    datajoint \
    imageio \
    plotly \
    GitPython \
    urllib3 \
    deeplake[video] \
    scikit-image \
    && python -m ipykernel.kernelspec
# deeplake to scikit-image added by Nicholas


# pytorch
RUN pip --no-cache-dir install \
    torch \
    torchvision \
    torchaudio \
    --extra-index-url https://download.pytorch.org/whl/cu116

# Nicholas: commenting this out because it takes,, forever,, and is not used in the sensorium code I'm pretty sure
# jax
# RUN apt install g++
# RUN git clone https://github.com/google/jax \
#     && cd jax \
#     && python build/build.py --enable_cuda --cuda_version="11.6" --cudnn_version="8.4" \
#     && pip install dist/*.whl\
#     && pip install -e .


# enable jupyter extensions
RUN pip install jupyter_contrib_nbextensions \
    && jupyter contrib nbextension install --system \
    && jupyter nbextension enable codefolding/main \
    && jupyter nbextension enable hinterland/hinterland \
    && jupyter nbextension enable varInspector/main \
    && jupyter nbextension enable comment-uncomment/main

# I'm not using R
# add R kernel
# RUN conda update conda
# RUN conda init zsh && exec zsh
# RUN conda install -c r r-essentials rstudio -y 
# RUN conda install -c conda-forge r-irkernel -y
# RUN R -e 'IRkernel::installspec(user = FALSE)'

# installing neuralpredictors locally instead of from remote so i don't have to push every edit
# RUN python -m pip install -e git+https://github.com/nwcimaszewski/neuralpredictors
COPY ./sinz_repos/neuralpredictors /neuralpredictors
RUN pip install -e /neuralpredictors 

RUN python -m pip install git+https://github.com/sinzlab/nnfabrik
# RUN python -m pip install git+https://github.com/sinzlab/sensorium

# this I can probably also clone inside the image but I am just copying the local repo for now
COPY dlgn_cnn/docker_scripts/docker_scripts/sensorium_2023/ /sensorium_2023
RUN pip install -e /sensorium_2023 

RUN pip --no-cache-dir install scikit-image

# Jupyter has issues with being run directly:
#   https://github.com/ipython/ipython/issues/7062
# We just add a little wrapper script. #
COPY dlgn_cnn/docker_scripts/run_jupyterlab.sh /usr/local/bin
RUN chmod -R a+rwx /usr/.jupyter \
    && chmod +rx /usr/local/bin/run_jupyterlab.sh

USER $NB_USER

CMD ["/usr/local/bin/run_jupyterlab.sh"]
