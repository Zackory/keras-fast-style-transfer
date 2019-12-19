# keras-fast-style-transfer
Fast style transfer in Keras

## Setup
```bash
python3 -m pip install --user virtualenv
python3 -m venv env
source env/bin/activate
pip3 install tensorflow keras
git clone https://github.com/Zackory/keras-fast-style-transfer.git
cd keras-fast-style-transfer
mkdir data
cd data
wget http://msvocds.blob.core.windows.net/coco2014/train2014.zip
unzip train2014.zip
```

## Train a new style transfer model
```bash
python3 main.py --checkpoint-dir models/ --style wave.jpg --test chicago.jpg --test-dir tests/ --checkpoint-iterations 5
```
