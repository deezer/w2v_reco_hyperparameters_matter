wget https://repo.continuum.io/archive/Anaconda2-4.2.0-Linux-x86_64.sh
#apt-get install bzip2
bash Anaconda2-4.2.0-Linux-x86_64.sh -b -p /root/anaconda
source ~/.bashrc

/root/anaconda/bin/conda update conda -y
/root/anaconda/bin/conda install accelerate -y

# kind of necessary for now
/root/anaconda/bin/pip install --upgrade pip llvmlite