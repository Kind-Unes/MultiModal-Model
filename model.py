import requests
import os
from dotenv import load_dotenv
import transformers

# load the env variables from .env
load_dotenv()

# access the API_TOKENS
txt2img_api_token = os.getenv("API_TOKEN")



headers = {"Authorization": "Bearer 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"}

def query(payload):
	response = requests.post(txt2img_api_token, headers=headers, json=payload)
	return response.content
image_bytes = query({
	"inputs": "Astronaut riding a horse",
})
# You can access the image with PIL.Image for example
import io
from PIL import Image
image = Image.open(io.BytesIO(image_bytes))