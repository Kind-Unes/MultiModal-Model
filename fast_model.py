from config import * # import all variables and core functions
from fastapi import FastAPI, UploadFile, File , Form , HTTPException
from fastapi.responses import JSONResponse
import base64
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

# Other functions will be added soon