FROM debian:stretch

COPY anaconda_install.sh /srv/workspace/

WORKDIR /srv/workspace/

# Basic imports
RUN apt-get update && apt-get install -y \
  git \
  wget \
  tree \
  build-essential \
  vim \
  cron \
  nano

# Install anaconda - batch of scientific python computing tools
RUN /bin/bash /srv/workspace/anaconda_install.sh yes
RUN rm Anaconda2-4.2.0-Linux-x86_64.sh
RUN rm anaconda_install.sh

# and make specific installation path available.
ENV PATH=$PATH:/root/anaconda/bin/

# Install gensim
RUN pip install -U gensim
# solve issue with mkl (cf.: https://github.com/ContinuumIO/anaconda-issues/issues/698)
RUN conda install nomkl numpy scipy scikit-learn numexpr
# and copy modified word2vec.py file.
COPY word2vec.py /root/anaconda/lib/python2.7/site-packages/gensim/models/word2vec.py

# Copy src python code
COPY src /srv/workspace/src
# and add src to python path.
ENV PYTHONPATH=$PYTHONPATH:/srv/workspace/src

# Copy data
COPY data /srv/workspace/data

# Install other python packages
RUN pip install tqdm
RUN pip install prettytable


