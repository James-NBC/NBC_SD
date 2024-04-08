import requests
import argparse

API_NAME = "generate_image"

def parse_args():
    parser = argparse.ArgumentParser("Stable Diffusion Inference")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the server on")
    parser.add_argument("--prompt", type=str, default="A beautiful landscape painting", help="Prompt for the model")
    parser.add_argument("--output-path", type=str, default="output.jpg", help="Path to save the generated image")
    return parser.parse_args()

def main():
    args = parse_args()
    port = args.port
    prompt = args.prompt
    output_path = args.output_path
    url = f"http://localhost:{port}/{API_NAME}"
    json_request = {
        "output_path": output_path,
        "prompt": prompt,
        "C": 4,
        "H": 512,
        "W": 512,
        "f": 8,
        "ddim_steps": 25,
        "n_samples": 1,
        "scale": 7.5,
    }
    response = requests.post(url, json=json_request)
    print(response.status_code)
    print(response.text)

if __name__ == "__main__":
    main()