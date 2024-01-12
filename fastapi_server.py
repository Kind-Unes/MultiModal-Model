from config import * # import all variables and core functions
from fastapi import FastAPI, UploadFile, File , Form , HTTPException
from fastapi.responses import JSONResponse
import base64
import json
app = FastAPI()

# =================================================================================================
# Model: gemini-vision-pro
# Task: Visual Q&A
# function: answers any question from image(s) input(s)
# =================================================================================================
@app.post("/image_to_text/gemini_vision")
async def image_to_text_gemini_vision(images: UploadFile = File(...),prompt: str = Form(...),role : str = File(...)):
    try:
        data = {"prompt":prompt,"role":role}
        
        result = await core.gemini_img2txt(data,images)
        
        return JSONResponse(content={"status":"success","result":result})
    except Exception as e:
        raise HTTPException(status_code=500,detail={"status":"error","message":str(e)})

# =================================================================================================
# Model: gemini-pro
# Task: All NLP tasks
# function: Chatgpt alternative
# =================================================================================================
@app.post("/text_generation/gemini")
async def text_generation(role: str = Form(...),prompt: str = Form(...)):
    try:
        result = core.text_generation(role,prompt)

        return JSONResponse(content={"status":"success","result":result})
    except Exception as e:
        raise HTTPException(status_code=500,detail={"status":"error","message":str(e)})

# =================================================================================================
# Model: ssd-1b-anime
# Task: text to image
# function: generates anime images from text input
# =================================================================================================
@app.post("/text_to_image/ssd_1b_anime")
async def text_to_image_ssd_1b_anime(prompt: str = Form(...)):
    try:
        image_bytes = core.text2image(prompt, txt2img_SDD_1B_ANIME_api_token)

        if MODEL_LOADING_MESSAGE.encode() in image_bytes:
            raise HTTPException(status_code=401, detail=MODEL_LOADING_MESSAGE)

        result = base64.b64encode(image_bytes).decode("utf-8")

        return JSONResponse(content={"status": "success", "result": result})
    except Exception as e:
        raise HTTPException(status_code=500, detail={"status": "error", "message": str(e)})

# =================================================================================================
# Model : SSD_1B (Base Model)
# Speciality: Base Model
# Prompt : {"prompt":"NEON CAT"}
# =================================================================================================
@app.post("/txt2img/SSD_1B")
async def text2image_SSD_1B(prompt: str):
    try:
        image_bytes = core.text2image({"inputs": prompt}, txt2img_SDD_1B_api_token)
        base64_encoded_image = base64.b64encode(image_bytes).decode('utf-8')
        return JSONResponse(content={"status": "success", "image": base64_encoded_image})
    except Exception as e:
        raise HTTPException(status_code=500, detail={"status": "error", "message": str(e)})

# =================================================================================================
# Model : OPENDALLE-V1 (Base Model)
# Speciality: Base Model
# Prompt : {"prompt":"NEON CAT"}
# =================================================================================================
@app.post("/txt2img/OPENDALLE")
async def text2image_OPENDALLE(prompt: str):
    try:
        image_bytes = core.text2image({"inputs": prompt}, txt2img_OPENDALLE_api_token)
        base64_encoded_image = base64.b64encode(image_bytes).decode('utf-8')
        return JSONResponse(content={"status": "success", "image": base64_encoded_image})
    except Exception as e:
        raise HTTPException(status_code=500, detail={"status": "error", "message": str(e)})

# =================================================================================================
# Model : OPENDALLE-V1 (EXAMPLE)
# Speciality: Generates Content based on this prompt: Ultra Realistic, Neon Lightning , 16K, Face Focus , Anime Picture , Smooth Lightning 
# Prompt : {"prompt":"NEON CAT"}
# =================================================================================================
@app.post("/txt2img/OPENDALLE/Speciality")
async def text2image_OPENDALLE_Speciality_1(prompt: str):
    try:
        image_bytes = core.text2image({"inputs": prompt}, txt2img_OPENDALLE_api_token)
        base64_encoded_image = base64.b64encode(image_bytes).decode('utf-8')
        return JSONResponse(content={"status": "success", "image": base64_encoded_image})
    except Exception as e:
        raise HTTPException(status_code=500, detail={"status": "error", "message": str(e)})

# =================================================================================================
# Model : BLIP_IMAGE_CAPTIONING_LARGE 
# Speciality: Base Model
# prompt : {"file":file}
# =================================================================================================
@app.post('/img2txt/BLIP_IMAGE_CAPTIONING_LARGE')
async def image2text_BLIP(file: UploadFile = File(...)):
    try:
        data = await file.read()
        result = core.image2text(data, img2txt_BLIP_api_token)
        return JSONResponse(content={"status": "success", "text": result[0]["generated_text"]})
    except Exception as e:
        raise HTTPException(status_code=500, detail={"status": "error", "message": str(e)})

# =================================================================================================
# Model : BLIP_IMAGE_CAPTIONING_LARGE 
# Speciality: Base Model
# prompt : {"file":file}
# =================================================================================================
@app.post('/audio2txt/WHISPER_LARGE_V2')
async def audio2text_WHISPER(file: UploadFile = File(...)):
    try:
        data = await file.read()
        output = core.audio2text(data)
        return JSONResponse(content={"status": "success", "text": output})
    except Exception as e:
        raise HTTPException(status_code=500, detail={"status": "error", "message": str(e)})

# =================================================================================================
# Model : MMS_TTS_ENGs 
# Speciality: Base Model
# prompt : {"prompt":"hello world!"}
# =================================================================================================
@app.post('/txt2audio/MMS_TTS_ENG')
async def text2audio_MMS_TSS_ENG(data: dict):
    try:
        text_input = data.get('prompt')
        audio_bytes = core.text2audio(text_input, txt2audio_MMS_TTS_ENG_api_token)
        return JSONResponse(content={'audio': base64.b64encode(audio_bytes).decode('utf-8')})
    except Exception as e:
        raise HTTPException(status_code=500, detail={'status': 'error', 'message': str(e)})
    
    
# =================================================================================================
# IMAGE TO AUDIO
# =================================================================================================
def query_huggingface_api(data):
    response = requests.post(img2txt_BLIP_api_token, headers=headers, data=data)
    return response.json()

@app.post('/img2audio/MMS_BLIP')
async def image2audio_MMS_BLIP(file: UploadFile = File(...)):
    try:
        data = await file.read()
        result = query_huggingface_api(data)

        audio_bytes = core.text2audio(result[0]["generated_text"], txt2audio_MMS_TTS_ENG_api_token)
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')

        return JSONResponse(content={"status": "success", "audio": audio_base64}, status_code=200)
    except Exception as e:
        raise HTTPException(status_code=500, detail={"status": "error", "message": str(e)})

# =================================================================================================
# AUDIO TO IMAGE
# =================================================================================================
@app.post('/audio2img/WHISPER_OPENDALLE')
async def audio2img_WHISPER_OPENDALLE(file: UploadFile = File(...)):
    try:
        text = core.audio2text(await file.read())

        image_bytes = core.text2image({
            "inputs": text,
        }, txt2img_OPENDALLE_api_token)

        base64_encoded_image = base64.b64encode(image_bytes).decode('utf-8')

        return JSONResponse(content={"status": "success", "Image": base64_encoded_image}, status_code=200)

    except Exception as e:
        raise HTTPException(status_code=500, detail={"status": "error", "message": str(e)})

@app.post('/audio2img/WHISPER_SSD_1B')
async def audio2img_WHISPER_SSD_1B(file: UploadFile = File(...)):
    try:
        text = core.audio2text(await file.read())

        image_bytes = core.text2image({
            "inputs": text,
        }, txt2img_SDD_1B_api_token)

        base64_encoded_image = base64.b64encode(image_bytes).decode('utf-8')

        return JSONResponse(content={"status": "success", "Image": base64_encoded_image}, status_code=200)

    except Exception as e:
        raise HTTPException(status_code=500, detail={"status": "error", "message": str(e)})

@app.post('/audio2img/WHISPER_SSD_1B_ANIME')
async def audio2img_WHISPER_SSD_1B_ANIME(file: UploadFile = File(...)):
    try:
        text = core.audio2text(await file.read())

        image_bytes = core.text2image({
            "inputs": text,
        }, txt2img_SDD_1B_ANIME_api_token)

        base64_encoded_image = base64.b64encode(image_bytes).decode('utf-8')

        return JSONResponse(content={"status": "success", "Image": base64_encoded_image}, status_code=200)

    except Exception as e:
        raise HTTPException(status_code=500, detail={"status": "error", "message": str(e)})

# =================================================================================================
# IMAGE CLASSIFICATION
# =================================================================================================
@app.post('/image_classification/RESNET')
async def image_classification_RESNET(file: UploadFile = File(...)):
    try:
        image = await file.read()
        result = core.image_classification(image, img_classification_RESNET_api_token)
        parsed_result = json.loads(result)

        return JSONResponse(content={"status": "success", "classes": parsed_result}, status_code=200)

    except Exception as e:
        raise HTTPException(status_code=500, detail={"status": "error", "message": str(e)})

@app.post('/image_classification/VIT_AGE')
async def image_classification_VIT_AGE(file: UploadFile = File(...)):
    try:
        image = await file.read()
        result = core.image_classification(image, img_classification_VIT_AGE_api_token)
        parsed_result = json.loads(result)

        return JSONResponse(content={"status": "success", "classes": parsed_result}, status_code=200)

    except Exception as e:
        raise HTTPException(status_code=500, detail={"status": "error", "message": str(e)})
    

    
# =================================================================================================
# IMAGE CLASSIFICATION - NFWS
# =================================================================================================
@app.post('/image_classification/NFWS')
async def image_classification_NFWS(file: UploadFile = File(...)):
    try:
        image = await file.read()
        result = core.image_classification(image, img_classification_NFWS_api_token)
        parsed_result = json.loads(result)

        return JSONResponse(content={"status": "success", "classes": parsed_result}, status_code=200)

    except Exception as e:
        raise HTTPException(status_code=500, detail={"status": "error", "message": str(e)})

# =================================================================================================
# IMAGE SEGMENTATION - B2_CLOTHES
# =================================================================================================
@app.post("/image_segmentation/B2_CLOTHES")
async def image_segmentation_B2_CLOTHES(file: UploadFile = File(...)):
    try:
        data = await file.read()
        image_bytes = core.image_segmentation(data, img_segmentation_b2_clothes_api_token)

        # Parse the JSON string into a Python list of dictionaries
        data_list = json.loads(image_bytes)

        scores = []
        labels = []
        segmented_pictures = []

        for obj in data_list:
            score = obj['score']
            label = obj['label']
            mask_base64 = obj['mask']

            # DO APPEND STUFF :D
            scores.append(score)
            labels.append(label)
            segmented_pictures.append(mask_base64)

        return JSONResponse(content={"status": "success", "Segmented Image": {"labels": labels, "scores": scores,
                                                                             "segmented_pictures": segmented_pictures}},
                            status_code=200)

    except Exception as e:
        raise HTTPException(status_code=500, detail={"status": "error", "message": str(e)})

# =================================================================================================
# AUDIO CLASSIFICATION - Hubert_emotion
# =================================================================================================
@app.post('/image_classification/Hubert_emotion')
async def audio_classification_Hubert_emotion(file: UploadFile = File(...)):
    try:
        data = core.audio_classification(await file.read(), audio_classification_Hubert_emotion_api_token)

        list_data = json.loads(data)

        scores = []
        labels = []
        for obj in list_data:
            scores.append(obj["score"])
            labels.append(obj["label"])

        return JSONResponse(content={"status": "success", "result": {"scores": scores, "labels": labels}}, status_code=200)

    except Exception as e:
        raise HTTPException(status_code=500, detail={"status": "error", "message": str(e)})

# =================================================================================================
# AUDIO CLASSIFICATION - wav2vec2_lg_xlsr_en
# =================================================================================================
@app.post('/image_classification/wav2vec2_lg_xlsr_en')
async def audio_classification_wav2vec2_lg_xlsr_en(file: UploadFile = File(...)):
    try:
        data = core.audio_classification(await file.read(), audio_classification_wav2vec2_lg_xlsr_en_api_token)

        list_data = json.loads(data)

        scores = []
        labels = []
        for obj in list_data:
            scores.append(obj["score"])
            labels.append(obj["label"])

        return JSONResponse(content={"status": "success", "result": {"scores": scores, "labels": labels}}, status_code=200)

    except Exception as e:
        raise HTTPException(status_code=500, detail={"status": "error", "message": str(e)})

# =================================================================================================
# AUDIO CLASSIFICATION - distil_ast_audioset
# =================================================================================================
@app.post('/image_classification/distil_ast_audioset')
async def audio_classification_distil_ast_audioset(file: UploadFile = File(...)):
    try:
        data = core.audio_classification(await file.read(), audio_classification_distil_ast_audioset_api_token)

        list_data = json.loads(data)

        scores = []
        labels = []
        for obj in list_data:
            scores.append(obj["score"])
            labels.append(obj["label"])

        return JSONResponse(content={"status": "success", "result": {"scores": scores, "labels": labels}}, status_code=200)

    except Exception as e:
        raise HTTPException(status_code=500, detail={"status": "error", "message": str(e)})
    
    
# =================================================================================================
# IMAGE CLASSIFICATION - wav2vec2_large_xlsr_53_gender
# =================================================================================================
@app.post('/image_classification/wav2vec2_large_xlsr_53_gender')
async def audio_classification_wav2vec2_large_xlsr_53_gender(file: UploadFile = File(...)):
    try:
        data = core.audio_classification(await file.read(), audio_classification_wav2vec2_large_xlsr_53_gender_api_token)

        list_data = json.loads(data)

        scores = []
        labels = []
        for obj in list_data:
            scores.append(obj["score"])
            labels.append(obj["label"])

        return JSONResponse(content={"status": "success", "result": {"scores": scores, "labels": labels}}, status_code=200)

    except Exception as e:
        raise HTTPException(status_code=500, detail={"status": "error", "message": str(e)})

# =================================================================================================
# IMAGE CLASSIFICATION - mms_lid_126
# =================================================================================================
@app.post('/image_classification/mms_lid_126')
async def audio_classification_mms_lid_126(file: UploadFile = File(...)):
    try:
        data = core.audio_classification(await file.read(), audio_classification_mms_lid_126_api_token)

        list_data = json.loads(data)

        scores = []
        labels = []
        for obj in list_data:
            scores.append(obj["score"])
            labels.append(obj["label"])

        return JSONResponse(content={"status": "success", "result": {"scores": scores, "labels": labels}}, status_code=200)

    except Exception as e:
        raise HTTPException(status_code=500, detail={"status": "error", "message": str(e)})


# =================================================================================================
# OBJECT DETECTION - detr_resnet_50
# =================================================================================================
@app.post("/object_detection/detr_resnet_50")
async def object_detection_detr_resnet_50(file: UploadFile = File(...)):
    try:
        data = await file.read()
        image_bytes = core.object_detection(data, object_detection_detr_resnet_50_api_token)

        # Parse the JSON string into a Python list of dictionaries
        data_list = json.loads(image_bytes)
        print(data_list)

        scores = []
        labels = []
        segmented_pictures = []

        for obj in data_list:
            score = obj['score']
            label = obj['label']
            mask_base64 = obj['box']

            # DO APPEND STUFF :D
            scores.append(score)
            labels.append(label)
            segmented_pictures.append(mask_base64)

        return JSONResponse(content={"status": "success", "Segmented Image": {"labels": labels, "scores": scores,
                                                                             "segmented_pictures": segmented_pictures}},
                            status_code=200)

    except Exception as e:
        raise HTTPException(status_code=500, detail={"status": "error", "message": str(e)})


# =================================================================================================
# IMAGE TO IMAGE - gemini_opendalle
# =================================================================================================
@app.post("/image_to_image/gemini_opendalle/V1")
async def image_to_image_gemini_opendalleV1(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    try:
        img1 = await file1.read()
        img2 = await file2.read()

        data = {"prompt": "i want you to give me the descripition of how would be the output image if we merged these two pictures toghther,you are not including all details please include them all and describe this image for me including all details as an example (gender . . .) and then MERGE THEM !!",
                "role": "you are a professional image describer that gives all details about the input image in 500 words always"}
        imgs_data = [img1, img2]

        # implement gemmini vision
        prompt = core.gemini_img2txt(data, imgs_data)

        # transform the prompt into an image using gemini
        image_bytes = core.text2image(prompt, txt2img_OPENDALLE_api_token)

        # base64 Transformation
        base64_encoded_image = base64.b64encode(image_bytes).decode('utf-8')
        return JSONResponse(content={"status": "success", "Image": base64_encoded_image}, status_code=200)

    except Exception as e:
        raise HTTPException(status_code=500, detail={"status": "error", "message": str(e)})


# =================================================================================================
# EMOJI TO IMG - OPENDALLE_gemni
# =================================================================================================
@app.post("/emoji_to_image/OPENDALLE_gemni")
async def emoji_to_image():
    try:
        form = 0
        emoji = await form.parse_form('prompt')
        prompt = core.text_generation(emoji + " in one word what does emoji represent ?",
                                      "ultra-realistic,16k,smooth,focus,super resolution,high-quality")
        image_bytes = core.text2image(emoji + prompt, txt2img_OPENDALLE_api_token)
        base64_encoded_image = base64.b64encode(image_bytes).decode('utf-8')

        return JSONResponse(content={"status": "success", "image": base64_encoded_image}, status_code=200)

    except Exception as e:
        raise HTTPException(status_code=500, detail={"status": "error", "message": str(e)})


# =================================================================================================
# IMG TO EMOJI - gemini
# =================================================================================================
@app.post('/image_to_emoji/gemini')
async def image_to_emoji(file: UploadFile = File(...)):
    try:
        image = await file.read()
        data = {"prompt": "transform this image into emojis, you can include more than a single emoji",
                "role": "you should only use emojis no words are allowed"}
        result = core.gemini_img2txt(data, [image])

        return JSONResponse(content={"status": "success", "text": result}, status_code=200)
    except Exception as e:
        raise HTTPException(status_code=500, detail={"status": "error", "message": str(e)})

