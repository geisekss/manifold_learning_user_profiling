FROM tensorflow/tensorflow:1.9.0-gpu
ARG DEBIAN_FRONTEND=noninteractive
# install packages
RUN apt-get update -qq \
 && apt-get -y upgrade \
 && apt-get install apt-utils 
RUN apt-get install --no-install-recommends -y \
    # install essentials
    build-essential \
    man-db \
    wget \
    less \
    git \
    htop \
    vim \
    emacs \
    cmake \
    tmux \
    # install python 3
    python3 \
    python3-dev \
    python3-pip \
    python3-setuptools \
    python3-virtualenv \
    python3-colorama \
    python3-wheel \
    python3-tk \
    pkg-config \
    # requirements for numpy
    libopenblas-base \
    # for setting up locale (https://stackoverflow.com/q/39760663/968131)
    locales \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip
RUN pip3 install --no-cache-dir Cython

RUN pip3 install ipython numpy matplotlib h5py tqdm urllib3 scipy==1.1.0 scikit-learn scikit-image 
RUN pip3 install tensorflow-gpu==1.9.0 tflearn keras==2.2.2
