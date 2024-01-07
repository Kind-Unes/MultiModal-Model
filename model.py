import requests
import google.generativeai as genai 
import os
from dotenv import load_dotenv
from PIL import Image
from flask import Flask, jsonify, request
from flask_cors import CORS
import base64
from diffusers import DiffusionPipeline
import io

app = Flask(__name__)
CORS(app)

# Load the environment variables from .env
load_dotenv()

# Access the API_TOKENS
txt2img_api_token = os.getenv("API_TOKEN_SSD_1B_ANIME")
img2txt_api_token = os.getenv("API_TOKEN_BLIP_IMAGE_CAPTIONING_LARGE")
txt2audio_api_token = os.getenv("API_TOKEN_FACEBOOK_MMS_TTS_ENG")
audio2txt_api_token = os.getenv("API_TOKEN_OPENAI_WHISPER_LARGE_V2")
img_id_api_token = os.getenv("API_TOKEN_IMG_ID")
gemini_text_generation_api_token = os.getenv("API_TOKEN_GOOGLE_AI_STUDIO")


# stuff
authorization = os.getenv("HEADER_AUTH")
headers = {"Authorization": authorization}


#! ------------------------------------------------------------------------------------------------------
#!                                  # CORE FUNCTIONS 
#! ------------------------------------------------------------------------------------------------------


#? ------------------------------------------------------------------------------------------------------
#?                                  # Text Generation
#? ------------------------------------------------------------------------------------------------------

def txt_generation(role, prompt):
    prompt_parts = [
        role, prompt
    ]

    response = model.generate_content(prompt_parts)

    return response.text

#? ------------------------------------------------------------------------------------------------------
#?                                  # IMAGE TO TEXT
#? ------------------------------------------------------------------------------------------------------

def gemini_img2txt(data, image_file):
    try:
        model = genai.GenerativeModel('gemini-pro-vision')

        role = data["role"]
        prompt = data["prompt"]

        # Process the file as needed (e.g., convert to PIL.Image)
        image = Image.open(io.BytesIO(image_file.read()))

        response = model.generate_content([role + prompt, image], stream=False)

        return response.text

    except Exception as e:
        raise RuntimeError(f"Error generating vision image: {str(e)}")
    


def img2txt(data):
    response = requests.post(img2txt_api_token, headers=headers, data=data)
    return response.json()



#? ------------------------------------------------------------------------------------------------------
#?                                  # TEXT TO IMAGE
#? ------------------------------------------------------------------------------------------------------


def text2image(payload):
    response = requests.post(txt2img_api_token, headers=headers, json=payload)
    return response.content

#? ------------------------------------------------------------------------------------------------------
#?                                  # AUDIO TO TEXT
#? ------------------------------------------------------------------------------------------------------


def audio2text(audio_data):
    response = requests.post(audio2txt_api_token, headers=headers, files={"file": audio_data})
    return response.json()


#? ------------------------------------------------------------------------------------------------------
#?                                  # TEXT TO AUDIO
#? ------------------------------------------------------------------------------------------------------

def text2audio(text):
    payload = {"inputs": text}
    response = requests.post(txt2audio_api_token, headers=headers, json=payload)
    return response.content


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




@app.route('/text_generation/gemini', methods=['POST'])
def generate_text():
        
    try:
        data = request.json
        role = data["role"]
        prompt = data["prompt"]



        result = txt_generation(role,prompt)
   
        return jsonify({"status": "success", "result": result})

    except Exception as e:
        return jsonify({"status": "error", "message": f"Error generating text: {str(e)}"}),500
    


#! ------------------------------------------------------------------------------------------------------
#!                                  # Gemini Vision
#! ------------------------------------------------------------------------------------------------------

@app.route('/image2txt/gemini_vision', methods=['POST'])
def gemini_img2txt():
    try:
        if 'images' in request.files and 'prompt' in request.form and 'role' in request.form:
            image_file = request.files['images']
            prompt = request.form['prompt']
            role = request.form['role']

            data = {"prompt": prompt, "role": role}

            app.logger.info(f"Received data: {data}")

            result = gemini_img2txt(data, image_file)

            return jsonify({"status": "success", "result": result})
        else:
            return jsonify({"status": "error", "message": "Invalid request format"}), 400

    except Exception as e:
        app.logger.error(f"Error processing request: {str(e)}")
        return jsonify({"status": "error", "message": f"Error generating vision image: {str(e)}"}), 500


#! ------------------------------------------------------------------------------------------------------
#!                                  # TEXT TO IMAGE
#! ------------------------------------------------------------------------------------------------------


@app.route("/txt2img/SSD_1B_ANIME", methods=["POST"]) 
def generate_image():
    try:
        data = request.get_json()
        user_input = data.get("prompt")  # you can include a default prompt

        image_bytes = text2image({
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



@app.route('/img2txt/BLIP_IMAGE_CAPTIONING_LARGE', methods=['POST'])
def process_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        try:
            data = file.read()
            result = img2txt(data)
            return jsonify({"status": "success", "text": result[0]["generated_text"]})
        except Exception as e:
            return jsonify({"status": "error", "message": f"Error generating image: {str(e)}"}),500



#! ------------------------------------------------------------------------------------------------------
#!                                    # AUDIO TO TEXT 
#! ------------------------------------------------------------------------------------------------------

@app.route('/audio2txt/WHISPER_LARGE_V2', methods=['GET', 'POST'])
def audio2text():
    if request.method == 'POST':
        try:
            if 'file' not in request.files:
                return jsonify({'error': 'No file part'}), 400

            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': 'No selected file'}), 400

            output = audio2text(file)

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

@app.route('/txt2audio/MMS_TTS_ENG', methods=['POST'])
def text2audio():
        try:
            data = request.get_json()
            text_input = data.get('text', 'Hello, World!')

            audio_bytes = text2audio(text_input)

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

@app.route('/img2audio/MMS__BLIP', methods=['POST'])
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

            audio_bytes = text2audio(result[0]["generated_text"])
            audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')


            return jsonify({"status": "success", "audio": audio_base64})
        except Exception as e:
            print(e)
            return jsonify({"status": "error", "message": f"Error generating audio: {str(e)}"}),500



#! ------------------------------------------------------------------------------------------------------
#!                                     # Audio to Image
#! ------------------------------------------------------------------------------------------------------
        

@app.route('/audio2img/WHISPER__BLIP', methods=['GET', 'POST'])
def audio2img():
    if request.method == 'POST':
        try:
            if 'file' not in request.files:
                return jsonify({'error': 'No file part'}), 400

            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': 'No selected file'}), 400

            text = audio2text(file)


            
            image_bytes = text2image({
               "inputs": text,
             })
            
            base64_encoded_image = base64.b64encode(image_bytes).decode('utf-8')

            return jsonify({"status": "success", "Image": base64_encoded_image})

           
        except Exception as e:
            print(e)
            return jsonify({'error': str(e)}), 500


#! ------------------------------------------------------------------------------------------------------
#!                                    # IMAGE CLASSIFICATION
#! ------------------------------------------------------------------------------------------------------
    
#! ------------------------------------------------------------------------------------------------------
#!           # IMAGE TO IMAGE (USING DIFFUSERS + USING IMG =) TEXT(PROMPT ENGINEERING) =) IMAGE)
#! ------------------------------------------------------------------------------------------------------
     
#! ------------------------------------------------------------------------------------------------------
#!                                    # IMAGE SEGMENTATION
#! ------------------------------------------------------------------------------------------------------
    
#! ------------------------------------------------------------------------------------------------------
#!                                    # VIDEO CLASSIFICATION    
#! ------------------------------------------------------------------------------------------------------
    
#! ------------------------------------------------------------------------------------------------------
#!                                    # AUDIO TASKS
#! ------------------------------------------------------------------------------------------------------

#! ------------------------------------------------------------------------------------------------------
#!                                    # DEPTH ESTIMATION
#! ------------------------------------------------------------------------------------------------------
    
#! ------------------------------------------------------------------------------------------------------
#!                              # VISUAL Q&A (GOOGLE GEMINI)(PROMPT ENGINEERED) + HF.CO 
#! ------------------------------------------------------------------------------------------------------

#! ------------------------------------------------------------------------------------------------------
#!                                    #TEXT TO 3D (GIF)
#! ------------------------------------------------------------------------------------------------------
    
    
    
    



# Launch your server
if __name__ == '__main__':
    app.run(debug=True)















#!                                     OUT OF THE SERVICE        
""""
# ------------------------------------------------------------------------------------------------------
#                                    # Image ID Diffuser
#                                  API NOT WORKING CURRENTLY
# ------------------------------------------------------------------------------------------------------
     
def query(payload):
    response = requests.post(img_id_api_token, headers=headers, json=payload)
    return response.content

def image_to_base64(image_bytes):
    return base64.b64encode(image_bytes).decode('utf-8')

@app.route("/img2img/face_id", methods=["POST"])
def identify_face():
    try:
        # Get image file from the request
        image_file = request.files.get("image")
        
        if image_file is None:
            return jsonify({"status": "error", "message": "No image file provided"}), 400

        # Convert image file to bytes
        image_bytes = image_file.read()

        # Convert image bytes to Base64
        base64_encoded_image = image_to_base64(image_bytes)

        # Make the API request
        response_content = query({"inputs": base64_encoded_image})

        # Process the API response (you may need to adapt this based on the actual response format)
        result = {"status": "success", "response_content": response_content.decode("utf-8")}

        return jsonify(result)

    except Exception as e:
        return jsonify({"status": "error", "message": f"Error processing request: {str(e)}"}), 500

"""