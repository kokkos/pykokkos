FROM ubuntu

ENV HOME_DIR /home/pk

# Install base utilities.
RUN apt-get update \
    && apt-get install -y build-essential \
    && apt-get install -y wget \
    && apt-get install -y git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# https://stackoverflow.com/questions/27701930/how-to-add-users-to-docker-container
RUN useradd --create-home --shell /bin/bash pk
USER pk
WORKDIR $HOME_DIR

# Install miniconda.
ENV CONDA_DIR $HOME_DIR/opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
/bin/bash ~/miniconda.sh -b -p $HOME_DIR/opt/conda

# Put conda on the path so we can use the command.
ENV PATH=$CONDA_DIR/bin:$PATH

# Get pykokkos.
RUN git clone https://github.com/kokkos/pykokkos-base

# Install dependencies.
RUN conda install -c conda-forge --yes mamba
RUN conda init
RUN mamba install -c conda-forge --yes --file pykokkos-base/requirements.txt python=3.11
COPY requirements.txt /tmp/requirements.txt
RUN mamba install -c conda-forge --yes --file /tmp/requirements.txt
RUN cd pykokkos-base && python setup.py install -- -DENABLE_LAYOUTS=ON -DENABLE_MEMORY_TRAITS=OFF -DENABLE_VIEW_RANKS=3 -DENABLE_CUDA=OFF -DENABLE_THREADS=OFF -DENABLE_OPENMP=ON
