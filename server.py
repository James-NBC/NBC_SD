import argparse, os
import cv2
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from imwatermark import WatermarkEncoder
from itertools import islice
from einops import rearrange
import time
import onnxruntime
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import nullcontext

from ldm.util import instantiate_from_config
from ldm.models.diffusion.plms import PLMSSampler

from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import AutoFeatureExtractor
from flask import Flask, request, jsonify


app = Flask(__name__)

def load_model_from_config(config, ckpt, verbose=False):
    if ckpt.endswith(".ckpt"):
        print(f"Loading model from {ckpt}")
        pl_sd = torch.load(ckpt, map_location="cpu")
        if "global_step" in pl_sd:
            print(f"Global Step: {pl_sd['global_step']}")
        sd = pl_sd["state_dict"]
        model = instantiate_from_config(config.model)
        m, u = model.load_state_dict(sd, strict=False)
        if len(m) > 0 and verbose:
            print("missing keys:")
            print(m)
        if len(u) > 0 and verbose:
            print("unexpected keys:")
            print(u)
    elif ckpt.endswith(".safetensors"):
        print(f"Loading model from {ckpt}")
        from safetensors.torch import load_file
        sd = load_file(ckpt)
        model = instantiate_from_config(config.model)
        m, u = model.load_state_dict(sd, strict=False)
        if len(m) > 0 and verbose:
            print("missing keys:")
            print(m)
        if len(u) > 0 and verbose:
            print("unexpected keys:")
            print(u)
    else:
        raise ValueError(f"Unknown extension for checkpoint {ckpt}")

    model.cuda()
    model.eval()
    return model

def put_watermark(img, wm_encoder=None):
    if wm_encoder is not None:
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = wm_encoder.encode(img, 'dwtDct')
        img = Image.fromarray(img[:, :, ::-1])
    return img


def load_replacement(x):
    try:
        hwc = x.shape
        y = Image.open("assets/rick.jpeg").convert("RGB").resize((hwc[1], hwc[0]))
        y = (np.array(y)/255.0).astype(x.dtype)
        assert y.shape == x.shape
        return y
    except Exception:
        return x
    
def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images

def check_safety(x_image):
    safety_checker_input = safety_feature_extractor(numpy_to_pil(x_image), return_tensors="pt")
    x_checked_image, has_nsfw_concept = safety_checker(images=x_image, clip_input=safety_checker_input.pixel_values)
    assert x_checked_image.shape[0] == len(has_nsfw_concept)
    for i in range(len(has_nsfw_concept)):
        if has_nsfw_concept[i]:
            x_checked_image[i] = load_replacement(x_checked_image[i])
    return x_checked_image, has_nsfw_concept


def load_stable_diffusion(config_path, ckpt_path):
    # if not os.path.exists(ckpt_path):
    #     print("Downloading model checkpoint...")
    #     os.system(f"wget https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.ckpt?download=true -O {ckpt_path}")
    config = OmegaConf.load(config_path)
    model = load_model_from_config(config, ckpt_path)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    sampler = PLMSSampler(model)
    model.to(device)
    return model, sampler

def parse_args():
    parser = argparse.ArgumentParser("Stable Diffusion Inference")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the server on")
    parser.add_argument("--ckpt", type=str, required = True, help="Path to the model checkpoint")
    parser.add_argument("--verifier", type=str, required = True, help="Path to the verifier model")
    return parser.parse_args()

args = parse_args()
safety_model_id = "CompVis/stable-diffusion-safety-checker"
safety_feature_extractor = AutoFeatureExtractor.from_pretrained(safety_model_id)
safety_checker = StableDiffusionSafetyChecker.from_pretrained(safety_model_id)
wm = "StableDiffusionV1"
wm_encoder = WatermarkEncoder()
wm_encoder.set_watermark('bytes', wm.encode('utf-8'))
verifier = onnxruntime.InferenceSession(args.verifier)
model, sampler = load_stable_diffusion("inference_config.yaml", args.ckpt)

@app.route('/')
def index():
    return "Hello, World!"

@app.route('/generate_image', methods=['POST'])
def generate_image():
    seed_everything(42)
    json_request = request.get_json(force=True)
    output_path = json_request['output_path']
    prompt = json_request['prompt']
    requested_C = json_request['C']
    requested_H = json_request['H']
    requested_W = json_request['W']
    requested_f = json_request['f']
    requested_ddim_steps = json_request['ddim_steps']
    requested_scale = json_request['scale']
    start = time.time()
    start_code = None
    precision_scope = autocast
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                for prompts in tqdm([[prompt]], desc="data"):
                    uc = None
                    if requested_scale != 1.0:
                        uc = model.get_learned_conditioning(1 * [""])
                    if isinstance(prompts, tuple):
                        prompts = list(prompts)
                    c = model.get_learned_conditioning(prompts)
                    shape = [requested_C, requested_H // requested_f, requested_W // requested_f]
                    samples_ddim, _ = sampler.sample(S=requested_ddim_steps,
                                                        conditioning=c,
                                                        batch_size=1,
                                                        shape=shape,
                                                        verbose=False,
                                                        unconditional_guidance_scale=requested_scale,
                                                        unconditional_conditioning=uc,
                                                        eta=0.0,
                                                        x_T=start_code)

                    x_samples_ddim = model.decode_first_stage(samples_ddim)
                    x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                    x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()

                    x_checked_image, has_nsfw_concept = check_safety(x_samples_ddim)

                    x_checked_image_torch = torch.from_numpy(x_checked_image).permute(0, 3, 1, 2)
                    x_checked_image_numpy = x_checked_image_torch.cpu().numpy()
                    print(x_checked_image_numpy.shape)
                    verified_embedding = verifier.run(None, {'input': x_checked_image_numpy.astype(np.float32)})[0].tolist()
                    print(len(verified_embedding))

                    x_sample = 255. * rearrange(x_checked_image_numpy[0], 'c h w -> h w c')
                    img = Image.fromarray(x_sample.astype(np.uint8))
                    img = put_watermark(img, wm_encoder)
                    img.save(output_path)
                            
    return jsonify({"output_path": output_path, "embedding": verified_embedding, "time": time.time() - start})  

if __name__ == "__main__":
    app.run(host='127.0.0.1', port = args.port)