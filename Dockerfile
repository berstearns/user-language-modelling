FROM ubuntu:18.04
RUN apt update \
    && apt -y upgrade \
    && apt install -y python3-pip \
    && apt install -y build-essential libssl-dev libffi-dev python3-dev \ 
    && pip3 install torch==1.7.1+cpu torchvision==0.8.2+cpu torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html \  
    && pip3 install torchtext \
    && pip3 install -U pip setuptools wheel \
    && pip3 install -U spacy \
    && export LC_ALL=C.UTF-8 \
    && python3 -m spacy download en_core_web_sm \
    && python3 -m spacy download en_core_web_trf
