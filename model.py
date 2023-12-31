import requests
import os
from dotenv import load_dotenv
import io
from PIL import Image
from flask import Flask, jsonify, request
from flask_cors import CORS
import base64


app = Flask(__name__)
CORS(app)

# Load the environment variables from .env
load_dotenv()

# Access the API_TOKENS
txt2img_api_token = os.getenv("API_TOKEN_SSD_1B_ANIME")
authorization = os.getenv("HEADER_AUTH")
headers = {"Authorization": authorization}



# TEXT TO IMAGE
def query(payload):
    response = requests.post(txt2img_api_token, headers=headers, json=payload)
    return response.content

@app.route("/txt2img", methods=["POST"]) 
def generate_image():
    try:
        data = request.get_json()
        user_input = data.get("prompt")  # you can include a default prompt

        image_bytes = query({
            "inputs": user_input,
        })
        # server api testing
        print("Debugging : ---------------------------------------------- ",image_bytes)

        # Convert binary data to Base64-encoded string
        base64_encoded_image = base64.b64encode(image_bytes).decode('utf-8')

        # Return a successful response
        return jsonify({"status": "success", "image": base64_encoded_image})

    except Exception as e:
        # Return an error response if an exception occurs
        return jsonify({"status": "error", "message": f"Error generating image: {str(e)}"})


# IMAGE TO TEXT 





if __name__ == '__main__':
    app.run(debug=True)
