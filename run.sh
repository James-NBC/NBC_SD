conda env create -f environment.yaml
conda init bash
source activate NBC_SD
# variable port
export PORT=5000
python txt2img.py --prompt "a photograph of an astronaut riding a horse" --plms --ckpt sd_ckpt/model.ckpt --config configs/v1-inference.yaml --n_samples 1 --n_iter 1