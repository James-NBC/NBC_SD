conda env create -f environment.yaml
conda init bash
source activate NBC_SD
# variable port
export PORT=5000
python server.py --port $PORT
python client.py --port $PORT --prompt "a cat"