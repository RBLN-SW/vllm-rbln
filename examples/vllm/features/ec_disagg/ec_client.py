# Copyright 2026 Rebellions Inc. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Send one image question through the EPD proxy and print the answer.

The proxy has the producer encode the image and the consumer answer with the
encoder output it reads from the producer.

Usage:
    python ec_client.py --image path/to/image.jpg
    python ec_client.py --image https://example.com/image.jpg --prompt "..."
"""

import argparse
import base64
import mimetypes
from pathlib import Path

from openai import OpenAI


def image_url(image: str) -> str:
    if image.startswith(("http://", "https://", "data:")):
        return image
    mime = mimetypes.guess_type(image)[0] or "image/jpeg"
    data = base64.b64encode(Path(image).read_bytes()).decode()
    return f"data:{mime};base64,{data}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--image", required=True, help="Local path or URL.")
    parser.add_argument("--prompt", default="What is in this image?")
    parser.add_argument("--max-tokens", type=int, default=64)
    args = parser.parse_args()

    client = OpenAI(base_url=args.base_url, api_key="EMPTY")
    model = client.models.list().data[0].id
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url(args.image)}},
                    {"type": "text", "text": args.prompt},
                ],
            }
        ],
        max_tokens=args.max_tokens,
        temperature=0,
    )
    print(response.choices[0].message.content)


if __name__ == "__main__":
    main()
