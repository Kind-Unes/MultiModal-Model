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
img2txt_api_token = os.getenv("API_TOKEN_BLIP_IMAGE_CAPTIONING_LARGE")
txt2audio_api_token = os.getenv("API_TOKEN_FACEBOOK_MMS_TTS_ENG")
audio2txt_api_token = os.getenv("API_TOKEN_OPENAI_WHISPER_LARGE_V2")
gemini_text_generation_api_token = os.getenv("API_TOKEN_GOOGLE_AI_STUDIO")

# stuff
authorization = os.getenv("HEADER_AUTH")
headers = {"Authorization": authorization}

#! ------------------------------------------------------------------------------------------------------
#!                                  # Text GENERATION
#! ------------------------------------------------------------------------------------------------------

 
genai.configure(api_key=gemini_text_generation_api_token)

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

#! ------------------------------------------------------------------------------------------------------
#!                                  # TEXT TO IMAGE
#! ------------------------------------------------------------------------------------------------------

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

        # Convert binary data to Base64-encoded string
        base64_encoded_image = base64.b64encode(image_bytes).decode('utf-8')

        # Return a successful response
        return jsonify({"status": "success", "image": base64_encoded_image})

    except Exception as e:
        # Return an error response if an exception occurs
        return jsonify({"status": "error", "message": f"Error generating image: {str(e)}"}),500

#! ------------------------------------------------------------------------------------------------------
#!                                    # IMAGE TO TEXT 
#! ------------------------------------------------------------------------------------------------------


def query_huggingface_api(data):
    response = requests.post(img2txt_api_token, headers=headers, data=data)
    return response.json()

@app.route('/process_image', methods=['POST'])
def process_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        try:
            data = file.read()
            result = query_huggingface_api(data)
            return jsonify({"status": "success", "text": result[0]["generated_text"]})
        except Exception as e:
            return jsonify({"status": "error", "message": f"Error generating image: {str(e)}"}),500



#! ------------------------------------------------------------------------------------------------------
#!                                    # AUDIO TO TEXT 
#! ------------------------------------------------------------------------------------------------------


def query_audio_model(audio_data):
    response = requests.post(audio2txt_api_token, headers=headers, files={"file": audio_data})
    return response.json()

@app.route('/automatic_speech_recognation', methods=['GET', 'POST'])
def audio2text():
    if request.method == 'POST':
        try:
            if 'file' not in request.files:
                return jsonify({'error': 'No file part'}), 400

            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': 'No selected file'}), 400

            output = query_audio_model(file)

            if 'error' in output:
                return jsonify({'error': output['error']}), 400
            else:
                return jsonify({'result': output})

        except Exception as e:
            print(e)
            return jsonify({'error': str(e)}), 500

    

#! ------------------------------------------------------------------------------------------------------
#!                                    # Text to audio
#! ------------------------------------------------------------------------------------------------------




def query_tts_model(text):
    payload = {"inputs": text}
    response = requests.post(txt2audio_api_token, headers=headers, json=payload)
    return response.content

@app.route('/txt2audio', methods=['POST'])
def text2audio():
        try:
            data = request.get_json()
            text_input = data.get('text', 'Hello, World!')

            audio_bytes = query_tts_model(text_input)

            return jsonify({'audio': base64.b64encode(audio_bytes).decode('utf-8')})

        except Exception as e:
            print(e)
            return jsonify({'error': str(e)}), 500



#! ------------------------------------------------------------------------------------------------------
#!                                     # Image to Audio
#! ------------------------------------------------------------------------------------------------------
        
def query_huggingface_api(data):
    response = requests.post(img2txt_api_token, headers=headers, data=data)
    return response.json()

@app.route('/img2audio', methods=['POST'])
def img2audio():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        try:
            data = file.read()
            result = query_huggingface_api(data)

            audio_bytes = query_tts_model(result[0]["generated_text"])
            audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')


            return jsonify({"status": "success", "audio": audio_base64})
        except Exception as e:
            print(e)
            return jsonify({"status": "error", "message": f"Error generating audio: {str(e)}"}),500



#! ------------------------------------------------------------------------------------------------------
#!                                     # Audio to Image
#! ------------------------------------------------------------------------------------------------------
        

@app.route('/audio2img', methods=['GET', 'POST'])
def audio2img():
    if request.method == 'POST':
        try:
            if 'file' not in request.files:
                return jsonify({'error': 'No file part'}), 400

            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': 'No selected file'}), 400

            text = query_audio_model(file)


            
            image_bytes = query({
               "inputs": text,
             })
            
            base64_encoded_image = base64.b64encode(image_bytes).decode('utf-8')

            return jsonify({"status": "success", "Image": base64_encoded_image})

           
        except Exception as e:
            print(e)
            return jsonify({'error': str(e)}), 500

    





# Launch your server
if __name__ == '__main__':
    app.run(debug=True)
