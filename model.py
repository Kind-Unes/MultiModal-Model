from config import *
from flask import Flask, jsonify, request
import google.generativeai as genai 
from dotenv import load_dotenv
from flask_cors import CORS
from PIL import Image
from io import BytesIO
import requests
import base64
import json # for image classification outputg
import os
import io

# Define your flask server
app = Flask()

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


        result = core.text_generation(role,prompt)
   
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

            result = core.gemini_img2txt(data, image_file)

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

        image_bytes = core.text2image({
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

        image_bytes = core.text2image({
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
        

        image_bytes = core.text2image({
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

        image_bytes = core.text2image({
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
            result = core.image2text(data,img2txt_BLIP_api_token)
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

            output = core.audio2text(file)
    
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

            audio_bytes = core.text2audio(text_input,txt2audio_MMS_TTS_ENG_api_token)

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

            audio_bytes = core.text2audio(result[0]["generated_text"],txt2audio_MMS_TTS_ENG_api_token)
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

            text = core.audio2text(file)


            
            image_bytes = core.text2image({
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

            text = core.audio2text(file)


            
            image_bytes = core.text2image({
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

            text = core.audio2text(file)
            
            image_bytes = core.text2image({
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
                result = core.image_classification(image,img_classification_RESNET_api_token)
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
                result = core.image_classification(image,img_classification_VIT_AGE_api_token)
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
                result = core.image_classification(image,img_classification_NFWS_api_token)
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
            image_bytes = core.image_segmentation(data,img_segmentation_b2_clothes_api_token)

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
                data = core.audio_classification(file,audio_classification_Hubert_emotion_api_token)

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
                data = core.audio_classification(file,audio_classification_wav2vec2_lg_xlsr_en_api_token)

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
                data = core.audio_classification(file,audio_classification_distil_ast_audioset_api_token)

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
                data = core.audio_classification(file,audio_classification_wav2vec2_large_xlsr_53_gender_api_token)

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
                data = core.audio_classification(file,audio_classification_mms_lid_126_api_token)

                list_data = json.loads(data)

                scores = []
                labels = []
                for obj in list_data:
                    scores.append(obj["score"])
                    labels.append(obj["label"])

                return jsonify({"status":"success","result":{"scores":scores,"labels":labels}}),200
            
            except Exception as e:
                return jsonify({"status":"error","message":str(e)}),500

#! ------------------------------------------------------------------------------------------------------
#!                                    # OBJECT DETECTION
#! ------------------------------------------------------------------------------------------------------

# =================================================================================================
# Models : detr_resnet (base Model) 
# Speciality: Base Model
# prompt : {"file":file}
# =================================================================================================
@app.route("/object_detection/detr_resnet_50",methods=["POST"])
def object_detection_detr_resnet():
    if "file" not in request.files:
        return jsonify({"status":"error","message":"No File Part"}),400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"status":"error","message":"No Selected Files"}),400
    if file:
        try:
            data = file.read()
            image_bytes = core.object_detection(data,object_detection_detr_resnet_50_api_token)

            
            # Parse the JSON string into a Python list of dictionaries
            data_list = json.loads(image_bytes)
            print(data_list)

            scores =[]
            labels = []
            segmented_pictures=[]

            for obj in data_list:
                score = obj['score']
                label = obj['label']
                mask_base64 = obj['box']

                
                # DO APPEND STUFF :D
                scores.append(score)
                labels.append(label)
                segmented_pictures.append(mask_base64)

            return jsonify({"status":"sucess","Segmented Image":{"labels":labels,"scores":scores,"segemented_pictures":segmented_pictures}})
        except Exception as e:
            return jsonify({"status":"error","message":str(e)})



# =================================================================================================
# Models : yolos_fashionpedia (base Model) 
# Speciality: Base Model
# prompt : {"file":file}
# =================================================================================================
@app.route("/object_detection/yolos_fashionpedia",methods=["POST"])
def object_detection_yolos_fashionpedia():
    if "file" not in request.files:
        return jsonify({"status":"error","message":"No File Part"}),400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"status":"error","message":"No Selected Files"}),400
    if file:
        try:
            data = file.read()
            image_bytes = core.object_detection(data,object_detection_yolos_fashionpedia_api_token)

            
            # Parse the JSON string into a Python list of dictionaries
            data_list = json.loads(image_bytes)
            print(data_list)

            scores =[]
            labels = []
            segmented_pictures=[]

            for obj in data_list:
                score = obj['score']
                label = obj['label']
                mask_base64 = obj['box']

                
                # DO APPEND STUFF :D
                scores.append(score)
                labels.append(label)
                segmented_pictures.append(mask_base64)

            return jsonify({"status":"sucess","Segmented Image":{"labels":labels,"scores":scores,"segemented_pictures":segmented_pictures}})
        except Exception as e:
            return jsonify({"status":"error","message":str(e)})




# =================================================================================================
# Models : table_transformer_detection (base Model) 
# Speciality: Base Model
# prompt : {"file":file}
# =================================================================================================
@app.route("/object_detection/table_transformer_detection",methods=["POST"])
def object_detection_table_transformer_detection():
    if "file" not in request.files:
        return jsonify({"status":"error","message":"No File Part"}),400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"status":"error","message":"No Selected Files"}),400
    if file:
        try:
            data = file.read()
            image_bytes = core.object_detection(data,object_detection_table_transformer_detection_api_token)

            
            # Parse the JSON string into a Python list of dictionaries
            data_list = json.loads(image_bytes)
            print(data_list)

            scores =[]
            labels = []
            segmented_pictures=[]

            for obj in data_list:
                score = obj['score']
                label = obj['label']
                mask_base64 = obj['box']

                
                # DO APPEND STUFF :D
                scores.append(score)
                labels.append(label)
                segmented_pictures.append(mask_base64)

            return jsonify({"status":"sucess","Segmented Image":{"labels":labels,"scores":scores,"segemented_pictures":segmented_pictures}})
        except Exception as e:
            return jsonify({"status":"error","message":str(e)})



#! ------------------------------------------------------------------------------------------------------
#!                                      # IMAGE TO IMAGE 
#! ------------------------------------------------------------------------------------------------------

#?                           I T   N E E D S  M O R E  O F  P R O M P T  E N G I N E E R I N G

# =================================================================================================
# Models : Gemini + OPENDALLE (base Model) 
# Speciality: Base Models
# prompt : {"file":files}
# function : images merging  Version (1)
# =================================================================================================
@app.route("/image_to_image/gemini_opendalle/V1",methods=["POST"])
def image_to_image_gemini_opendalle():
    if "file1" not in request.files and "file2" not in request.files:
        return jsonify({"status":"error","message":"Your request is incomplete!"}),400
    file1 = request.files["file1"]
    file2 = request.files["file2"]

    if (file1.filename == "" and file2.filename == "") :
        return jsonify({"status":"error","message":"No Selected Image(s)"}),400
    try:    
     if file2 and file1:
        img1 = file1.read()
        img2 = file2. read()

        data = {"prompt": "i want you to give me the descripition of how would be the output image if we merged these two pictures toghther,you are not including all details please include them all and describe this image for me including all details as an example (gender . . .) and then MERGE THEM !!", "role": "you are a professional image describer that gives all details about the input image in 500 words always"}
        imgs_data = [img1,img2]

        # implement gemmini vision
        prompt = core.gemini_img2txt(data,imgs_data)

        # transform the prompt into an image using gemini
        image_bytes = core.text2image(prompt,txt2img_OPENDALLE_api_token)
        
        # base64 Transformation
        base64_encoded_image = base64.b64encode(image_bytes).decode('utf-8')
        return jsonify({"status": "success", "Image": base64_encoded_image})
        
    except Exception as e:
        return jsonify({"status":"error","message":str(e)})


#?                           I T   N E E D S  M O R E  O F  P R O M P T  E N G I N E E R I N G
#?                           I T   T A K E S  A  L O T  O F  T I M E

# =================================================================================================
# Models : Gemini + OPENDALLE (base Model) 
# Speciality: Base Models
# prompt : {"file":files}
# function : images merging  Version (2)
# =================================================================================================
@app.route("/image_to_image/gemini_opendalle/V2",methods=["POST"])
def image_to_image_gemini_opendalleV2():
    if "file1" not in request.files and "file2" not in request.files and "prompt" not in request.form and "role" not in request.form :
        return jsonify({"status":"error","message":"Your request is incomplete!"}),400
    file1 = request.files["file1"]
    file2 = request.files["file2"]

    if (file1.filename == "" and file2.filename == "") :
        return jsonify({"status":"error","message":"No Selected Image(s)"}),400
    try:    
     if file2 and file1:
        img1 = file1.read()
        img2 = file2.read()

        data = {"prompt": "describe this image for me including all details as an example (gender . . .)", "role": "you are a professional image describer that gives all details about the input image in 500 words always"}

        # implement gemmini vision
        prompt1 = core.gemini_img2txt(data,[img1])
        prompt2 = core.gemini_img2txt(data,[img2])

        # merging prompts
        txt_genertation_role = "you are a professional descriptions merging master"
        txt_generation_prompt = "i want you to merge these two descriptions and give me the description that would result if we merged these two descriptions"
        final_prompt = core.text_generation(txt_genertation_role,txt_generation_prompt)

        # transform the prompt into an image using gemini
        image_bytes = core.text2image(final_prompt,txt2img_OPENDALLE_api_token)
        
        # base64 Transformation
        base64_encoded_image = base64.b64encode(image_bytes).decode('utf-8')
        return jsonify({"status": "success", "Image": base64_encoded_image})
        
    except Exception as e:
        return jsonify({"status":"error","message":str(e)})

   
#! ------------------------------------------------------------------------------------------------------
#!                                    #EMOJI TO IMG
#! ------------------------------------------------------------------------------------------------------

# =================================================================================================
# Model : OPENDALLE-v1 + Gemini (Base Models)
# Speciality: Base Model
# Prompt : {"prompt":":Happy_face"}
# function : it generates images based on emotes only
# =================================================================================================
@app.route("/emoji_to_image/OPENDALLE_gemni", methods=["POST"]) 
def emotji_to_image():
    if "prompt" not in request.form :
        return jsonify({"status":"error","message":"Your request is incomplete!"}),400
    emoji = request.form["prompt"] 
    try:
        prompt = core.text_generation(emoji + " in one word what does emoji represent  ?","")
        
        image_generation_prompt = "ultra-realistic,16k,smooth,focus,super resolution,high-quality"       
        image_bytes = core.text2image(emoji+image_generation_prompt,txt2img_OPENDALLE_api_token)
        print(image_bytes)    
        base64_encoded_image = base64.b64encode(image_bytes).decode("UTF-8")


        return jsonify({"status": "success", "image": base64_encoded_image})

    except Exception as e:
        # Return an error response if an exception occurs
        return jsonify({"status": "error", "message": f"Error generating image: {str(e)}"}),500



#! ------------------------------------------------------------------------------------------------------
#!                                    #IMG TO EMOJI
#! ------------------------------------------------------------------------------------------------------


# =================================================================================================
# Model : Gemnin (base model) 
# Speciality: Base Model
# prompt : {"file":file}
# =================================================================================================
@app.route('/image_to_emoji/gemini', methods=['POST'])
def image_to_emoji():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        try:
            image = file.read()
            data = {"prompt" : "transform this image into emojis, you can include more then a single emoji " , "role":"you should only use emojis no words are allowed"}
            result = core.gemini_img2txt(data,[image])

            return jsonify({"status": "success", "text": result})
        except Exception as e:
            return jsonify({"status": "error", "message": f"Error generating emoji: {str(e)}"}),500




# Launch your server 
app.run(debug=True)









#?                                     COMING SOON . . .        

     
#! ------------------------------------------------------------------------------------------------------
#!                                    # VIDEO CLASSIFICATION    
#! ------------------------------------------------------------------------------------------------------

#! ------------------------------------------------------------------------------------------------------
#!                                    #TEXT TO 3D (GIF)
#! ------------------------------------------------------------------------------------------------------

#! ------------------------------------------------------------------------------------------------------
#!                                    #EMOJI TO 3D
#! ------------------------------------------------------------------------------------------------------











#?                                     OUT OF SERVICE        
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
