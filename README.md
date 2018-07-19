# Getting Started
This repository can be used to reproduce results of "Word2vec applied to Recommendation: Hyperparameters Matter" by H. Caselles-Dupré, F. Lesaint and J. Royo-Letelier. The paper will be published on the 12th ACM Conference on Recommender Systems, Vancouver, Canada, 2nd-7th October 2018

## Usage with Docker [recommended]

### Install

`git clone https://github.com/deezer/w2v_reco_hyperparameters_matter.git`

`cd w2v_reco_hyperparameters_matter`

`docker build -t w2v_reco_hyperparameters_matter .`

### Run

To reproduce results in *Table 1: Next Event Prediction*, line *Fully optimized SGNS* from paper:

`docker run -ti --name=music_1_sgns w2v_reco_hyperparameters_matter:latest /bin/bash -c "python src/main.py --path_data='data/music_1.npy' --p2v=1 --window_size=3 --it=110 --sample=0.00001 --power_alpha=-0.5"`

`docker run -ti --name=music_2_sgns w2v_reco_hyperparameters_matter:latest /bin/bash -c "python src/main.py --path_data='data/music_2.npy' --p2v=1 --window_size=3 --it=130 --sample=0.00001 --power_alpha=-0.5"`

`docker run -ti --name=ecommerce_sgns w2v_reco_hyperparameters_matter:latest /bin/bash -c "python src/main.py --path_data='data/ecommerce_sessions.npy' --p2v=1 --window_size=3 --it=140 --sample=0.001 --power_alpha=1"`

`docker run -ti --name=kosarak_sgns w2v_reco_hyperparameters_matter:latest /bin/bash -c "python src/main.py --path_data='data/kosarak_sessions.npy' --p2v=1 --window_size=7 --it=150 --sample=0.00001 --power_alpha=-1"`

To reproduce results in *Table 2: NEP performance in cold-start regime*, lines *Fully optimized MetaProd2vec* from paper:

`docker run -ti --name=music_1_cs_mp2v_0 w2v_reco_hyperparameters_matter:latest /bin/bash -c "python src/main.py --path_data='data/music_1.npy' --p2v=0 --window_size=7 --it=90 --sample=0.0001 --power_alpha=-0.5 --cold_start=0"`

`docker run -ti --name=music_1_cs_mp2v_2 w2v_reco_hyperparameters_matter:latest /bin/bash -c "python src/main.py --path_data='data/music_1.npy' --p2v=0 --window_size=7 --it=90  --sample=0.0001 --power_alpha=-0.5 --cold_start=2"`

`docker run -ti --name=music_2_cs_mp2v_0 w2v_reco_hyperparameters_matter:latest /bin/bash -c "python src/main.py --path_data='data/music_2.npy' --p2v=0 --window_size=3 --it=150 --sample=0.0001 --power_alpha=-0.5 --cold_start=0"`

`docker run -ti --name=music_2_cs_mp2v_2 w2v_reco_hyperparameters_matter:latest /bin/bash -c "python src/main.py --path_data='data/music_2.npy' --p2v=0 --window_size=3 --it=150 --sample=0.0001 --power_alpha=-0.5 --cold_start=2"`

## Usage without Docker

### Install

Install gensim locally following https://radimrehurek.com/gensim/install.html.

`git clone https://github.com/deezer/w2v_reco_hyperparameters_matter.git`

`cd w2v_reco_hyperparameters_matter`

Replace the file gensim/models/word2vec.py in your local gensim installation folder by word2vec.py.

`mkdir data`

Copy .npy files from https://drive.google.com/drive/folders/1S-vneh5-egjzjNP7y1ChdevOjukQopqX to the data folder.


### Run

To reproduce results in *Table 1: Next Event Prediction*, line *Fully optimized SGNS* from paper:

`python src/main.py --path_data='data/music_1.npy' --p2v=1 --window_size=3 --it=110 --sample=0.00001 --power_alpha=-0.5`

`python src/main.py --path_data='data/music_2.npy' --p2v=1 --window_size=3 --it=130 --sample=0.00001 --power_alpha=-0.5`

`python src/main.py --path_data='data/ecommerce_sessions.npy' --p2v=1 --window_size=3 --it=140 --sample=0.001 --power_alpha=1`

`python src/main.py --path_data='data/kosarak_sessions.npy' --p2v=1 --window_size=7 --it=150 --sample=0.00001 --power_alpha=-1`

To reproduce results in *Table 2: NEP performance in cold-start regime*, lines *Fully optimized MetaProd2vec* from paper:

`python src/main.py --path_data='data/music_1.npy' --p2v=0 --window_size=7 --it=90 --sample=0.0001 --power_alpha=-0.5 --cold_start=0`

`python src/main.py --path_data='data/music_1.npy' --p2v=0 --window_size=7 --it=90  --sample=0.0001 --power_alpha=-0.5 --cold_start=2`

`python src/main.py --path_data='data/music_2.npy' --p2v=0 --window_size=3 --it=150 --sample=0.0001 --power_alpha=-0.5 --cold_start=0`

`python src/main.py --path_data='data/music_2.npy' --p2v=0 --window_size=3 --it=150 --sample=0.0001 --power_alpha=-0.5 --cold_start=2`
