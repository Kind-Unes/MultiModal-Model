import google.generativeai as genai 
from dotenv import load_dotenv
import requests
from PIL import Image
import io
import os

# Load the enviroment variables
load_dotenv()

# Access the API_TOKENS
# Text To Image
txt2img_SDD_1B_ANIME_api_token = "https://api-inference.huggingface.co/models/furusu/SSD-1B-anime"
txt2img_SDD_1B_api_token = "https://api-inference.huggingface.co/models/segmind/SSD-1B"
txt2img_OPENDALLE_api_token = "https://api-inference.huggingface.co/models/dataautogpt3/OpenDalleV1.1"
img2txt_BLIP_api_token = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-large"

# Text To Audio
txt2audio_MMS_TTS_ENG_api_token = "https://api-inference.huggingface.co/models/facebook/mms-tts-eng"

# Audio To Text
audio2txt_WHISPER_api_token = "G E T  W H I S P H E R"

# Face ID 
img_id_api_token = "https://api-inference.huggingface.co/models/h94/IP-Adapter-FaceID"

# Text Generation | Image to text | Visuale Question & answering 
GEMINI_api_token = "AIzaSyAo8rIkKbS-bFHTYIUESCKBIBxYBl7la6U"

# IMAGE CLASSIFICATION
img_classification_RESNET_api_token = "https://api-inference.huggingface.co/models/microsoft/resnet-50"
img_classification_VIT_AGE_api_token = "https://api-inference.huggingface.co/models/nateraw/vit-age-classifier"
img_classification_NFWS_api_token= "https://api-inference.huggingface.co/models/Falconsai/nsfw_image_detection" # NOT SAFE TO WORK

# Image Segmentation 
img_segmentation_b2_clothes_api_token = "https://api-inference.huggingface.co/models/mattmdjaga/segformer_b2_clothes"

# Audio Classification 
audio_classification_Hubert_emotion_api_token= "https://api-inference.huggingface.co/models/Rajaram1996/Hubert_emotion"
audio_classification_wav2vec2_lg_xlsr_en_api_token = "https://api-inference.huggingface.co/models/ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
audio_classification_distil_ast_audioset_api_token = "https://api-inference.huggingface.co/models/bookbot/distil-ast-audioset"
audio_classification_wav2vec2_large_xlsr_53_gender_api_token = "https://api-inference.huggingface.co/models/alefiury/wav2vec2-large-xlsr-53-gender-recognition-librispeech"
audio_classification_mms_lid_126_api_token = "https://api-inference.huggingface.co/models/facebook/mms-lid-126"

# Object Detecton :
object_detection_detr_resnet_50_api_token = "https://api-inference.huggingface.co/models/facebook/detr-resnet-50"
object_detection_yolos_fashionpedia_api_token = "https://api-inference.huggingface.co/models/valentinafeve/yolos-fashionpedia"
object_detection_table_transformer_detection_api_token = "https://api-inference.huggingface.co/models/microsoft/table-transformer-detection"

# Others
authorization = os.getenv("HEADER_AUTH")
headers = {"Authorization": authorization}
genai.configure(api_key=GEMINI_api_token)

# Google Gemini paramaters
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

# Gemini Model initialization
model = genai.GenerativeModel(
model_name="gemini-pro",
generation_config=generation_config,
safety_settings=safety_settings,
    )
chat = model.start_chat(history=[])

# Core Functions
class Core:
    @staticmethod    
    def text_generation(role, prompt):
        prompt_parts = [
        role, prompt
         ]
        response = chat.send_message(prompt_parts)
        #response = model.generate_content(prompt_parts)
    
        return response.text

    @staticmethod
    def gemini_img2txt(data, image_array):
            model = genai.GenerativeModel('gemini-pro-vision')
            role = data["role"]
            prompt = data["prompt"]
            processed_images = []

            for image in image_array:
                image = Image.open(io.BytesIO(image))
                processed_images.append(image)
            
            data_array = [role + prompt] + processed_images
            response = model.generate_content(data_array, stream=False)
            return response.text

    @staticmethod
    def image2text(data,api):
        response = requests.post(api, headers=headers, data=data)
        return response.json()

    @staticmethod
    def text2image(payload,api):
        response = requests.post(api, headers=headers, json=payload)
        return response.content

    @staticmethod
    def audio2text(audio_data):
        response = requests.post(audio2txt_WHISPER_api_token, headers=headers, files={"file": audio_data})
        return response.json()


    @staticmethod
    def text2audio(text,api):
        payload = {"inputs": text}
        response = requests.post(api, headers=headers, json=payload)
        return response.content

    @staticmethod
    def image_classification(data,api):
        response = requests.post(api, headers=headers, data=data)
        return response.content

    @staticmethod
    def image_segmentation(data,api):
        response = requests.post(api,headers=headers,data=data)
        return response.content

    @staticmethod
    def audio_classification(data,api):
        response = requests.post(api,headers=headers,data=data)
        return response.content

    @staticmethod
    def object_detection(data,api):
        response = requests.post(api,headers=headers,data=data)
        return response.content
    

# creating a class instance
core = Core()


# Error Messages
MODEL_LOADING_MESSAGE = "is currently loading"

