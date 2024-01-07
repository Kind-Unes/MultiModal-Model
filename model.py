import requests
import google.generativeai as genai 
import os
from dotenv import load_dotenv
from PIL import Image
from flask import Flask, jsonify, request
from flask_cors import CORS
import base64
from diffusers import DiffusionPipeline
from io import BytesIO
import json # for image classification output
# TODO: API TOKEN SPECIALIZE THEM
# TODO: ADD MORE MODELS FOR EACH TASKS
# TODO: IMPLEMENT SPECIALIZED FUNCTIONS FOR EACH MODEL 
# TODO: IMPLEMENT THE REST OF THE TASKS
# TODO: FIX THE CORE FUNCTIONS AND UNITE THEM 
# TODO: INPUT JSON VERIFICATION
#


app = Flask(__name__)
CORS(app)

# Load the environment variables from .env
load_dotenv()

# Access the API_TOKENS
txt2img_SDD_1B_ANIME_api_token = os.getenv("SSD_1B_ANIME")
txt2img_SDD_1B_api_token = os.getenv("SSD_1B")
txt2img_OPENDALLE_api_token = os.getenv("OPENDALLE")
img2txt_BLIP_api_token = os.getenv("BLIP_IMAGE_CAPTIONING_LARGE")
txt2audio_MMS_TTS_ENG_api_token = os.getenv("FACEBOOK_MMS_TTS_ENG")
audio2txt_WHISPER_api_token = os.getenv("OPENAI_WHISPER_LARGE_V2")
img_id_api_token = os.getenv("IMG_ID")
GEMINI_api_token = os.getenv("GOOGLE_AI_STUDIO")

# IMAGE CLASSIFICATION
img_classification_RESNET_api_token = "https://api-inference.huggingface.co/models/microsoft/resnet-50"
img_classification_VIT_AGE_api_token = "https://api-inference.huggingface.co/models/nateraw/vit-age-classifier"
img_classification_NFWS_api_token= "https://api-inference.huggingface.co/models/Falconsai/nsfw_image_detection" # NOT SAFE TO WORK
#! GEMINI (ultimate one)

# Image Segmentation 
img_segmentation_b2_clothes_api_token = "https://api-inference.huggingface.co/models/mattmdjaga/segformer_b2_clothes"

# Audio Classification 
audio_classification_Hubert_emotion_api_token= "https://api-inference.huggingface.co/models/Rajaram1996/Hubert_emotion"
audio_classification_wav2vec2_lg_xlsr_en_api_token = "https://api-inference.huggingface.co/models/ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
audio_classification_distil_ast_audioset_api_token = "https://api-inference.huggingface.co/models/bookbot/distil-ast-audioset"
audio_classification_wav2vec2_large_xlsr_53_gender_api_token = "https://api-inference.huggingface.co/models/alefiury/wav2vec2-large-xlsr-53-gender-recognition-librispeech"
audio_classification_mms_lid_126_api_token = "https://api-inference.huggingface.co/models/facebook/mms-lid-126"





# stuff
authorization = os.getenv("HEADER_AUTH")
headers = {"Authorization": authorization}
genai.configure(api_key=GEMINI_api_token)


#! ------------------------------------------------------------------------------------------------------
#!                                  # CORE FUNCTIONS 
#! ------------------------------------------------------------------------------------------------------





#? ------------------------------------------------------------------------------------------------------
#?                                  # Text Generation
#? ------------------------------------------------------------------------------------------------------

def text_generation(role, prompt):
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

    prompt_parts = [
        role, prompt
    ]

    response = model.generate_content(prompt_parts)

    return response.text

#? ------------------------------------------------------------------------------------------------------
#?                                  # IMAGE TO TEXT
#? ------------------------------------------------------------------------------------------------------

# Model : Gemini
# Function : ?
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

# Model : Gemini
# Function : ?
def image2text(data):
    response = requests.post(img2txt_BLIP_api_token, headers=headers, data=data)
    return response.json()


#? ------------------------------------------------------------------------------------------------------
#?                                  # TEXT TO IMAGE
#? ------------------------------------------------------------------------------------------------------

def text2image(payload,api):
    response = requests.post(api, headers=headers, json=payload)
    return response.content

#? ------------------------------------------------------------------------------------------------------
#?                                  # AUDIO TO TEXT
#? ------------------------------------------------------------------------------------------------------

def audio2text(audio_data):
    response = requests.post(audio2txt_WHISPER_api_token, headers=headers, files={"file": audio_data})
    return response.json()


#? ------------------------------------------------------------------------------------------------------
#?                                  # TEXT TO AUDIO
#? ------------------------------------------------------------------------------------------------------

def text2audio(text,api):
    payload = {"inputs": text}
    response = requests.post(api, headers=headers, json=payload)
    return response.content


#? ------------------------------------------------------------------------------------------------------
#?                                  # IMAGE CLASSIFICATION
#? ------------------------------------------------------------------------------------------------------

def image_classification(data,api):
    response = requests.post(api, headers=headers, data=data)
    return response.content

#? ------------------------------------------------------------------------------------------------------
#?                                  # IMAGE SEGEMENTATION
#? ------------------------------------------------------------------------------------------------------

def image_segmentation(data,api):
    response = requests.post(api,headers=headers,data=data)
    return response.content
#? ------------------------------------------------------------------------------------------------------
#?                                  # IMAGE SEGEMENTATION
#? ------------------------------------------------------------------------------------------------------

def audio_classification(data,api):
    response = requests.post(api,headers=headers,data=data)
    return response.content
#! ------------------------------------------------------------------------------------------------------
#!                                  # Text GENERATION
#! ------------------------------------------------------------------------------------------------------

# =================================================================================================
# Model : gemini (Base Model)
# Speciality: Base Model
# prompt : {"prompt":"Say Unes is COooOoOoL"}
# =================================================================================================
@app.route('/text_generation/gemini', methods=['POST'])
def generate_text_gemini():
        
    try:
        data = request.json
        role = data["role"]
        prompt = data["prompt"]



        result = text_generation(role,prompt)
   
        return jsonify({"status": "success", "result": result})

    except Exception as e:
        return jsonify({"status": "error", "message": f"Error generating text: {str(e)}"}),500
    


#! ------------------------------------------------------------------------------------------------------
#!                                  # Gemini Vision
#! ------------------------------------------------------------------------------------------------------

# =================================================================================================
# Model : gemini_vision (Base Model)
# Speciality: Base Model
# prompt : {"prompt":"WHO IS THIS","role":"you are Unes Fan","images":[image1,Image2]}
# =================================================================================================
@app.route('/image2txt/gemini_vision', methods=['POST'])
def image2text_gemini():
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

# =================================================================================================
# Model : SSD_1B_ANIME (Base Model)
# Speciality: Base Model
# Prompt : {"prompt":"NEON CAT"}
# =================================================================================================
@app.route("/txt2img/SSD_1B_ANIME", methods=["POST"]) 
def text2image_SSD_1B_ANIME():
    try:
        data = request.get_json()
        user_input = data.get("prompt")  # you can include a default prompt

        image_bytes = text2image({
            "inputs": user_input,
        },txt2img_SDD_1B_ANIME_api_token)

        # Convert binary data to Base64-encoded string
        base64_encoded_image = base64.b64encode(image_bytes).decode('utf-8')

        # Return a successful response
        return jsonify({"status": "success", "image": base64_encoded_image})

    except Exception as e:
        # Return an error response if an exception occurs
        return jsonify({"status": "error", "message": f"Error generating image: {str(e)}"}),500

# =================================================================================================
# Model : SSD_1B (Base Model)
# Speciality: Base Model
# Prompt : {"prompt":"NEON CAT"}
# =================================================================================================
@app.route("/txt2img/SSD_1B", methods=["POST"]) 
def text2image_SSD_1B():
    try:
        data = request.get_json()
        user_input = data.get("prompt")  # you can include a default prompt

        image_bytes = text2image({
            "inputs": user_input,
        },txt2img_SDD_1B_api_token)

        # Convert binary data to Base64-encoded string
        base64_encoded_image = base64.b64encode(image_bytes).decode('utf-8')

        # Return a successful response
        return jsonify({"status": "success", "image": base64_encoded_image})

    except Exception as e:
        # Return an error response if an exception occurs
        return jsonify({"status": "error", "message": f"Error generating image: {str(e)}"}),500
    

# =================================================================================================
# Model : OPENDALLE-V1 (Base Model)
# Speciality: Base Model
# Prompt : {"prompt":"NEON CAT"}
# =================================================================================================
@app.route("/txt2img/OPENDALLE", methods=["POST"]) 
def text2image_OPENDALLE():
    try:
        data = request.get_json()
        user_input = data.get("prompt") 
        

        image_bytes = text2image({
            "inputs": user_input,
        },txt2img_OPENDALLE_api_token)

        # Convert binary data to Base64-encoded string
        base64_encoded_image = base64.b64encode(image_bytes).decode('utf-8')

        # Return a successful response
        return jsonify({"status": "success", "image": base64_encoded_image})

    except Exception as e:
        # Return an error response if an exception occurs
        return jsonify({"status": "error", "message": f"Error generating image: {str(e)}"}),500

# =================================================================================================
# Model : OPENDALLE-V1 (EXAMPLE)
# Speciality: Generates Content based on this prompt: Ultra Realistic, Neon Lightning , 16K, Face Focus , Anime Picture , Smooth Lightning 
# Prompt : {"prompt":"NEON CAT"}
# =================================================================================================
@app.route("/txt2img/OPENDALLE/Speciality", methods=["POST"]) 
def text2image_OPENDALLE_Speciality_1():
    try:
        data = request.get_json()
        user_input = data.get("prompt")  # you can include a default prompt

        image_bytes = text2image({
            "inputs": user_input,
        },txt2img_OPENDALLE_api_token)

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

# =================================================================================================
# Model : BLIP_IMAGE_CAPTIONING_LARGE 
# Speciality: Base Model
# prompt : {"file":file}
# =================================================================================================
@app.route('/img2txt/BLIP_IMAGE_CAPTIONING_LARGE', methods=['POST'])
def image2text_BLIP():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        try:
            data = file.read()
            result = image2text(data)
            return jsonify({"status": "success", "text": result[0]["generated_text"]})
        except Exception as e:
            return jsonify({"status": "error", "message": f"Error generating image: {str(e)}"}),500



#! ------------------------------------------------------------------------------------------------------
#!                                    # AUDIO TO TEXT 
#! ------------------------------------------------------------------------------------------------------
        
# =================================================================================================
# Model : BLIP_IMAGE_CAPTIONING_LARGE 
# Speciality: Base Model
# prompt : {"file":file}
# =================================================================================================
@app.route('/audio2txt/WHISPER_LARGE_V2', methods=['POST'])
def audio2text_WHISPER():
        try:
            if 'file' not in request.files:
                return jsonify({'error': 'No file part'}), 400

            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': 'No selected file'}), 400

            output = audio2text(file)
    
            return jsonify({"status": "success", "text":output})

        except Exception as e:
            print(e)
            return jsonify({"status": "error", "message": f"Error generating text: {str(e)}"}),500

    

#! ------------------------------------------------------------------------------------------------------
#!                                    # Text to audio
#! ------------------------------------------------------------------------------------------------------

# =================================================================================================
# Model : MMS_TTS_ENGs 
# Speciality: Base Model
# prompt : {"prompt":"hello world!"}
# =================================================================================================
@app.route('/txt2audio/MMS_TTS_ENG', methods=['POST'])
def text2audio_MMS_TSS_ENG():
        try:
            data = request.get_json()
            text_input = data.get('prompt')

            audio_bytes = text2audio(text_input,txt2audio_MMS_TTS_ENG_api_token)

            return jsonify({'audio': base64.b64encode(audio_bytes).decode('utf-8')})

        except Exception as e:
            print(e)
            return jsonify({'error': str(e)}), 500



#! ------------------------------------------------------------------------------------------------------
#!                                     # Image to Audio
#! ------------------------------------------------------------------------------------------------------
        

# =================================================================================================
# Models : BLIP_IMAGE_CAPTIONING_LARGE + MMS_TTS_ENG
# Speciality: Base Models
# prompt : {"file":file}
# =================================================================================================
def query_huggingface_api(data):
    response = requests.post(img2txt_BLIP_api_token, headers=headers, data=data)
    return response.json()

@app.route('/img2audio/MMS_BLIP', methods=['POST'])
def image2audio_MMS_BLIP():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        try:
            data = file.read()
            result = query_huggingface_api(data)

            audio_bytes = text2audio(result[0]["generated_text"],txt2audio_MMS_TTS_ENG_api_token)
            audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')


            return jsonify({"status": "success", "audio": audio_base64})
        except Exception as e:
            print(e)
            return jsonify({"status": "error", "message": f"Error generating audio: {str(e)}"}),500



#! ------------------------------------------------------------------------------------------------------
#!                                     # Audio to Image
#! ------------------------------------------------------------------------------------------------------
    
# =================================================================================================
# Models : WHISPER + OPENDALLE (base Model)
# Speciality: Base Models
# prompt : {"file":file}
# =================================================================================================
@app.route('/audio2img/WHISPER_OPENDALLE', methods=['GET', 'POST'])
def audio2img_WHISPER_OPENDALLE():
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
             },txt2img_OPENDALLE_api_token)
            
            base64_encoded_image = base64.b64encode(image_bytes).decode('utf-8')

            return jsonify({"status": "success", "Image": base64_encoded_image})

           
        except Exception as e:
            print(e)
            return jsonify({'error': str(e)}), 500

# =================================================================================================
# Models : WHISPER + SDD-1B (base Model)
# Speciality: Base Models
# prompt : {"file":file}
# =================================================================================================
@app.route('/audio2img/WHISPER_SSD_1B', methods=['GET', 'POST'])
def audio2img_WHISPER_SSD_1B():
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
             },txt2img_SDD_1B_api_token)
            
            base64_encoded_image = base64.b64encode(image_bytes).decode('utf-8')

            return jsonify({"status": "success", "Image": base64_encoded_image})

           
        except Exception as e:
            print(e)
            return jsonify({'error': str(e)}), 500
        

# =================================================================================================
# Models : WHISPER + SDD-1B-ANIME (base Model)
# Speciality: Base Models
# prompt : {"file":file}
# =================================================================================================
@app.route('/audio2img/WHISPER_SSD_1B_ANIME', methods=['POST'])
def audio2img_WHISPER_SSD_1B_ANIME():
        try:
            if 'file' not in request.files:
                return jsonify({'error': 'No file part'}), 400

            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': 'No selected file'}), 400

            text = audio2text(file)
            
            image_bytes = text2image({
               "inputs": text,
             },txt2img_SDD_1B_ANIME_api_token)
            
            base64_encoded_image = base64.b64encode(image_bytes).decode('utf-8')

            return jsonify({"status": "success", "Image": base64_encoded_image})

           
        except Exception as e:
            print(e)
            return jsonify({'error': str(e)}), 500


#! ------------------------------------------------------------------------------------------------------
#!                                    # IMAGE CLASSIFICATION
#! ------------------------------------------------------------------------------------------------------

# =================================================================================================
# Models : RESNET (base Model)
# Speciality: Base Model
# prompt : {"file":file}
# =================================================================================================
@app.route('/image_classification/RESNET',methods=["POST"])
def image_classification_RESNET():
        # Input Verification
        if 'file' not in request.files:
            return jsonify({"error":"No file part"}),400
        file = request.files["file"]

        if file.filename == "":
            return jsonify({"error":"No Selected File"}),400
        
        if file:
            try:
                image = file.read() 
                result = image_classification(image,img_classification_RESNET_api_token)
                parsed_result = json.loads(result)
                print(parsed_result)

                return jsonify({"status":"success","classes":parsed_result}),200
            
            except Exception as e:
                return jsonify({"status":"error","message":str(e)}),500
            

# =================================================================================================
# Models : VIT_AGE (base Model)
# Speciality: Base Model
# prompt : {"file":file}
# =================================================================================================
@app.route('/image_classification/VIT_AGE',methods=["POST"])
def image_classification_VIT_AGE():
        # Input Verification
        if 'file' not in request.files:
            return jsonify({"error":"No file part"}),400
        file = request.files["file"]

        if file.filename == "":
            return jsonify({"error":"No Selected File"}),400
        
        if file:
            try:
                image = file.read() 
                result = image_classification(image,img_classification_VIT_AGE_api_token)
                parsed_result = json.loads(result)
                print(parsed_result)

                return jsonify({"status":"success","classes":parsed_result}),200
            
            except Exception as e:
                return jsonify({"status":"error","message":str(e)}),500

# =================================================================================================
# Models : NFWS (base Model)
# Speciality: Base Model
# prompt : {"file":file}
# =================================================================================================
@app.route('/image_classification/NFWS',methods=["POST"])
def image_classification_NFWS():
        # Input Verification
        if 'file' not in request.files:
            return jsonify({"error":"No file part"}),400
        file = request.files["file"]

        if file.filename == "":
            return jsonify({"error":"No Selected File"}),400
        
        if file:
            try:
                image = file.read() 
                result = image_classification(image,img_classification_NFWS_api_token)
                parsed_result = json.loads(result)
                print(parsed_result)

                return jsonify({"status":"success","classes":parsed_result}),200
            
            except Exception as e:
                return jsonify({"status":"error","message":str(e)}),500


#! ------------------------------------------------------------------------------------------------------
#!                                    # IMAGE SEGMENTATION
#! ------------------------------------------------------------------------------------------------------

# =================================================================================================
# Models : B2_CLOTHES (base Model)
# Speciality: Base Model
# prompt : {"file":file}
# =================================================================================================

#? THERE IS AN ERROR IN THE SCORES (always set to one D:)
@app.route("/image_segmentation/B2_CLOTHES",methods=["POST"])
def image_segmentation_B2_CLOTHES():
    if "file" not in request.files:
        return jsonify({"status":"error","message":"No File Part"}),400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"status":"error","message":"No Selected Files"}),400
    if file:
        try:
            data = file.read()
            image_bytes = image_segmentation(data,img_segmentation_b2_clothes_api_token)

            # Parse the JSON string into a Python list of dictionaries
            data_list = json.loads(image_bytes)

            scores =[]
            labels = []
            segmented_pictures=[]

            for obj in data_list:
                score = obj['score']
                label = obj['label']
                mask_base64 = obj['mask']

                
                # DO APPEND STUFF :D
                scores.append(score)
                labels.append(label)
                segmented_pictures.append(mask_base64)

            return jsonify({"status":"sucess","Segmented Image":{"labels":labels,"scores":scores,"segemented_pictures":segmented_pictures}})
        except Exception as e:
            return jsonify({"status":"error","message":str(e)})


    
#! ------------------------------------------------------------------------------------------------------
#!                                    # AUDIO CLASSIFICATION 
#! ------------------------------------------------------------------------------------------------------

# =================================================================================================
# Models : Hubert_emotion (base Model)
# Speciality: Base Model
# prompt : {"file":file}
# =================================================================================================
@app.route('/image_classification/Hubert_emotion',methods=["POST"])
def audio_classification_Hubert_emotion():
        # Input Verification
        if 'file' not in request.files:
            return jsonify({"error":"No file part"}),400
        file = request.files["file"]

        if file.filename == "":
            return jsonify({"error":"No Selected File"}),400
        
        if file:
            try:
                data = audio_classification(file,audio_classification_Hubert_emotion_api_token)

                list_data = json.loads(data)

                scores = []
                labels = []
                for obj in list_data:
                    scores.append(obj["score"])
                    labels.append(obj["label"])

                return jsonify({"status":"success","result":{"scores":scores,"labels":labels}}),200
            
            except Exception as e:
                return jsonify({"status":"error","message":str(e)}),500



# =================================================================================================
# Models : wav2vec2_lg_xlsr_en (base Model) Emotions as well :D
# Speciality: Base Model
# prompt : {"file":file}
# =================================================================================================
@app.route('/image_classification/wav2vec2_lg_xlsr_en',methods=["POST"])
def audio_classification_wav2vec2_lg_xlsr_en():
        # Input Verification
        if 'file' not in request.files:
            return jsonify({"error":"No file part"}),400
        file = request.files["file"]

        if file.filename == "":
            return jsonify({"error":"No Selected File"}),400
        
        if file:
            try:
                data = audio_classification(file,audio_classification_wav2vec2_lg_xlsr_en_api_token)

                list_data = json.loads(data)

                scores = []
                labels = []
                for obj in list_data:
                    scores.append(obj["score"])
                    labels.append(obj["label"])

                return jsonify({"status":"success","result":{"scores":scores,"labels":labels}}),200
            
            except Exception as e:
                return jsonify({"status":"error","message":str(e)}),500



# =================================================================================================
# Models : distil_ast_audioset (base Model) IT Detects what the voice is !
# Speciality: Base Model
# prompt : {"file":file}
# =================================================================================================
@app.route('/image_classification/distil_ast_audioset',methods=["POST"])
def audio_classification_distil_ast_audioset():
        # Input Verification
        if 'file' not in request.files:
            return jsonify({"error":"No file part"}),400
        file = request.files["file"]

        if file.filename == "":
            return jsonify({"error":"No Selected File"}),400
        
        if file:
            try:
                data = audio_classification(file,audio_classification_distil_ast_audioset_api_token)

                list_data = json.loads(data)

                scores = []
                labels = []
                for obj in list_data:
                    scores.append(obj["score"])
                    labels.append(obj["label"])

                return jsonify({"status":"success","result":{"scores":scores,"labels":labels}}),200
            
            except Exception as e:
                return jsonify({"status":"error","message":str(e)}),500


# =================================================================================================
# Models : wav2vec2_large_xlsr_53_gender (base Model) 
# Speciality: Base Model
# prompt : {"file":file}
# =================================================================================================
@app.route('/image_classification/wav2vec2_large_xlsr_53_gender',methods=["POST"])
def audio_classification_wav2vec2_large_xlsr_53_gender():
        # Input Verification
        if 'file' not in request.files:
            return jsonify({"error":"No file part"}),400
        file = request.files["file"]

        if file.filename == "":
            return jsonify({"error":"No Selected File"}),400
        
        if file:
            try:
                data = audio_classification(file,audio_classification_wav2vec2_large_xlsr_53_gender_api_token)

                list_data = json.loads(data)

                scores = []
                labels = []
                for obj in list_data:
                    scores.append(obj["score"])
                    labels.append(obj["label"])

                return jsonify({"status":"success","result":{"scores":scores,"labels":labels}}),200
            
            except Exception as e:
                return jsonify({"status":"error","message":str(e)}),500



# =================================================================================================
# Models : mms_lid_126 (base Model) For Languages classification
# Speciality: Base Model
# prompt : {"file":file}
# =================================================================================================
@app.route('/image_classification/mms_lid_126',methods=["POST"])
def audio_classification_mms_lid_126():
        # Input Verification
        if 'file' not in request.files:
            return jsonify({"error":"No file part"}),400
        file = request.files["file"]

        if file.filename == "":
            return jsonify({"error":"No Selected File"}),400
        
        if file:
            try:
                data = audio_classification(file,audio_classification_mms_lid_126_api_token)

                list_data = json.loads(data)

                scores = []
                labels = []
                for obj in list_data:
                    scores.append(obj["score"])
                    labels.append(obj["label"])

                return jsonify({"status":"success","result":{"scores":scores,"labels":labels}}),200
            
            except Exception as e:
                return jsonify({"status":"error","message":str(e)}),500



# Launch your server
app.run(debug=True)

#?                                     COMING SOON . . .        


#! ------------------------------------------------------------------------------------------------------
#!           # IMAGE TO IMAGE (USING DIFFUSERS + USING IMG =) TEXT(PROMPT ENGINEERING) =) IMAGE)
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
    
#! ------------------------------------------------------------------------------------------------------
#!                                    #EMOJI TO IMG
#! ------------------------------------------------------------------------------------------------------

#! ------------------------------------------------------------------------------------------------------
#!                                    #EMOJI TO TEXT
#! ------------------------------------------------------------------------------------------------------

#! ------------------------------------------------------------------------------------------------------
#!                                    #EMOJI TO AUDIO
#! ------------------------------------------------------------------------------------------------------

#! ------------------------------------------------------------------------------------------------------
#!                                    #EMOJI TO 3D
#! ------------------------------------------------------------------------------------------------------

#! ------------------------------------------------------------------------------------------------------
#!                                    #IMG TO EMOJI
#! ------------------------------------------------------------------------------------------------------

#! ------------------------------------------------------------------------------------------------------
#!                                    #TEXT TO EMOJI
#! ------------------------------------------------------------------------------------------------------

#! ------------------------------------------------------------------------------------------------------
#!                                    #AUDIO TO EMOJI
#!                                  (VOICE SENTIMENT ANALYSIS) 
#! ------------------------------------------------------------------------------------------------------

#! ------------------------------------------------------------------------------------------------------
#!                                    #3D TO EMOJI
#!                                    (ADVANCED)
#! ------------------------------------------------------------------------------------------------------



#?                                     OUT OF THE SERVICE        
""""
# ------------------------------------------------------------------------------------------------------
#                                    # Image ID Diffuser
#                                  API IS NOT WORKING CURRENTLY
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