import requests
import google.generativeai as genai
import os
from dotenv import load_dotenv
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
GOOGLE_AI_STUDIO = os.getenv("API_TOKEN_GOOGLE_AI_STUDIO")


# Text Generation 
genai.configure(api_key=GOOGLE_AI_STUDIO)

# Set up the model
generation_config = {
    "temperature": 0.9,
    "top_p": 1,
    "top_k": 1,
    "max_output_tokens": 2048,
}

safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "block_none"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "block_none"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "block_none"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "block_none"},
]

model = genai.GenerativeModel(
    model_name="gemini-pro",
    generation_config=generation_config,
    safety_settings=safety_settings,
)



def generate_content(role, prompt):
    prompt_parts = [
        role, prompt
    ]

    response = model.generate_content(prompt_parts)

    return response.text


@app.route('/generate', methods=['POST'])
def generate_text():
        
    try:
        data = request.json
        role = data["role"]
        prompt = data["prompt"]



        result = generate_content(role,prompt)
   
        return jsonify({"status": "success", "result": result})

    except Exception as e:
        return jsonify({"status": "error", "message": f"Error generating text: {str(e)}"}),500



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
        return jsonify({"status": "error", "message": f"Error generating image: {str(e)}"}),500


# IMAGE TO TEXT 





if __name__ == '__main__':
    app.run(debug=True)
