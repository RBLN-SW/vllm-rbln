from io import BytesIO

import requests
from PIL import Image
from vllm import LLM, SamplingParams


DOG_URL = "https://upload.wikimedia.org/wikipedia/commons/2/26/YellowLabradorLooking_new.jpg"


def load_image(src: str) -> Image.Image:
    if src.startswith(("http://", "https://")):
        resp = requests.get(src, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
        resp.raise_for_status()
        image = Image.open(BytesIO(resp.content))
    else:
        image = Image.open(src)
    image = image.convert("RGB")
    image.thumbnail((1008, 1008))
    return image


def main() -> None:
    llm = LLM(
        model="nvidia/MiniMax-M3-NVFP4",
        model_impl="vllm",
        dtype="bfloat16",
        tensor_parallel_size=4,
        max_num_seqs=1,
        max_num_batched_tokens=512,
        block_size=1024,
        limit_mm_per_prompt={"image": 1, "video": 0},
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_pil", "image_pil": load_image(DOG_URL)},
                {"type": "text", "text": "What is in this image?"},
            ],
        }
    ]
    outputs = llm.chat(messages, SamplingParams(temperature=0, max_tokens=256))

    print(f"Response: {outputs[0].outputs[0].text}")



if __name__ == "__main__":
    main()
