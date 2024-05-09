import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from datetime import datetime
import torch
import cv2
from PIL import Image
from io import BytesIO
import base64
import traceback
import numpy as np
import os
import re
import glob
from models import load_textdetector_model, dispatch_textdetector

from manga_ocr import MangaOcr

use_cuda = torch.cuda.is_available()

mocr = MangaOcr()
load_textdetector_model(use_cuda)

def infer(img, foldername, filename, lang, tech):
    separator = '\n@@@@@-mangatool-@@@@@\n'
    re_str = r'\n@@@@@-mangatool-@@@@@\n'
    mask, mask_refined, blk_list = dispatch_textdetector(img, use_cuda)
    torch.cuda.empty_cache()

    mask = cv2.dilate((mask > 170).astype('uint8')*255, np.ones((5,5), np.uint8), iterations=5)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    filter_mask = np.zeros_like(mask)
    for i, blk in enumerate(blk_list):
        xmin, ymin, xmax, ymax = blk.xyxy
        xmin = 0 if xmin < 0 else xmin
        ymin = 0 if ymin < 0 else ymin
        xmax = img.shape[1] if xmax >  img.shape[1] else xmax
        ymax = img.shape[0] if ymax >  img.shape[0] else ymax
        filter_mask[int(ymin):int(ymax), int(xmin):int(xmax)] = 1

    bboxes = []
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for i, contour in enumerate(contours):
        bbox = cv2.boundingRect(contour)
        xmin, ymin, w, h = bbox
        xmax = xmin + w
        ymax = ymin + h
        xmin = 0 if xmin < 0 else xmin
        ymin = 0 if ymin < 0 else ymin
        xmax = img.shape[1] if xmax >  img.shape[1] else xmax
        ymax = img.shape[0] if ymax >  img.shape[0] else ymax
        # index = np.bincount(np.ravel(filter_mask[int(ymin):int(ymax), int(xmin):int(xmax)])).argmax()
        index = np.sum(filter_mask[int(ymin):int(ymax), int(xmin):int(xmax)])
        if index > 0:
            bboxes.append(list(bbox))
            filter_mask[int(ymin):int(ymax), int(xmin):int(xmax)] = 0

    texts = []
    for bbox in bboxes:
        xmin, ymin, w, h = bbox
        xmax = xmin + w
        ymax = ymin + h
        xmin = 0 if xmin < 0 else xmin
        ymin = 0 if ymin < 0 else ymin
        xmax = img.shape[1] if xmax >  img.shape[1] else xmax
        ymax = img.shape[0] if ymax >  img.shape[0] else ymax
        #IMPORTANT ===================================================================================
        text = mocr(Image.fromarray(img[int(ymin):int(ymax), int(xmin):int(xmax), :]))
        if use_cuda:
            torch.cuda.empty_cache()
        texts.append(text)

    frames = [[0, img.shape[0],int(img.shape[1]/2), img.shape[1]], [0, img.shape[0], 0, int(img.shape[1]/2)]]
    frame_img = np.zeros_like(mask)
    frame_boxes = []
    frame_texts = []

    for i, frame in enumerate(frames):
        ymin, ymax, xmin, xmax = frame
        xmin = 0 if xmin < 0 else xmin
        ymin = 0 if ymin < 0 else ymin
        xmax = img.shape[1] if xmax >  img.shape[1] else xmax
        ymax = img.shape[0] if ymax >  img.shape[0] else ymax
        frame_img[ymin: ymax, xmin:xmax] = i+1
        frame_boxes.append([])
        frame_texts.append([])

    for bbox, text in zip(bboxes,texts):
        xmin, ymin, w, h = bbox
        xmax = xmin + w
        ymax = ymin + h
        xmin = 0 if xmin < 0 else xmin
        ymin = 0 if ymin < 0 else ymin
        xmax = img.shape[1] if xmax >  img.shape[1] else xmax
        ymax = img.shape[0] if ymax >  img.shape[0] else ymax
        index = np.bincount(np.ravel(frame_img[int(ymin):int(ymax), int(xmin):int(xmax)])).argmax()
        if index > 0:
            frame_boxes[int(index-1)].append(bbox)
            frame_texts[int(index-1)].append(text)

    final_text = []
    final_bboxes = None
    for _bboxes, _texts in zip(frame_boxes, frame_texts):
        if len(_bboxes) != 0:
            a = np.array(_bboxes)
            arg =  np.argsort(a[:,1])
            # arg = np.argsort(img.shape[1] - (a[:,0] + a[:,2]))
            # arg1 =  np.argsort(a[:,1])
            # arg = np.argsort(np.argsort(arg)*np.argsort(arg1)*(img.shape[1] - a[:,0])*(a[:,1]))
            _texts = np.array(_texts)[arg.astype(int)]
            final_text.append(separator.join(_texts))
            if final_bboxes is None:
                final_bboxes = a[arg.astype(int)]
            else:
                final_bboxes = np.concatenate((final_bboxes, a[arg.astype(int)]))

    
    text = separator.join(final_text)
    text_ref = separator.join(final_text)
    text_ref = re.sub(re_str, '', text_ref)
    if not text_ref == "":
        if not os.path.exists('output/' + foldername + "/"):
            os.mkdir('output/' + foldername + "/")
        with open('output/' + foldername + "/" + filename.split('.')[0] + '.txt', 'w+', encoding="utf-8") as f:
            f.write(text)
        f.close()
        np.savetxt('output/' + foldername + '/' + filename.split('.')[0] + '_bbox.txt', final_bboxes.astype(int))
        np.savetxt('output/' + foldername + '/' + filename.split('.')[0] + '_order.txt', np.array(range(len(final_bboxes))).astype(int), fmt="%d")

    return text, np.array2string(final_bboxes, precision=2, separator=',')
        
def sub(img, foldername, filename, lang='jp', tech="MangaOCR"):
    img = cv2.cvtColor(np.array(img).astype('uint8'), cv2.COLOR_RGB2BGR)
    res =  infer(img, foldername, filename, lang, tech)
    return res


app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

async def read_image(request):
    form = await request.form()
    file = await form["file"].read()
    image = Image.open(BytesIO(file))
    return image

def img2str(result):
    _, buffer = cv2.imencode('.jpg', result)
    img_str = base64.b64encode(buffer)
    return img_str


@app.get("/")
async def read_index():
    return FileResponse('index.html')

@app.get("/text/{foldername}/{filename}")
async def text_file(foldername, filename):
    return FileResponse('output/' + foldername + "/" + filename + ".txt")

@app.get("/bbox/{foldername}/{filename}")
async def bbox_file(foldername, filename):
    return FileResponse('output/' + foldername + "/" + filename + "_bbox.txt")

@app.get("/order/{foldername}/{filename}")
async def order_file(foldername, filename):
    return FileResponse('output/' + foldername + "/" + filename + "_order.txt")

@app.get("/folderlist")
async def folderlist():
    lists = os.listdir("output")
    times = []
    counts = []
    for file in lists:
        counts.append(len(glob.glob("output/" + file + "/*_bbox.txt")))
        times.append(os.path.getmtime('output/' + file))
    return {"file_list": lists, "times": times, "counts": counts}

@app.post("/update/{foldername}/{filename}")
async def update_file(request: Request, foldername, filename):
    payload = await request.json()
    try:
        if 'order' in payload:
            with open('output/' + foldername + "/" + filename + '_order.txt', 'w+', encoding="utf-8") as f:
                f.write(payload['order'])
        if 'text' in payload:
            with open('output/' + foldername + "/" + filename + '.txt', 'w+', encoding="utf-8") as f:
                f.write(payload['text'])
        return {"message": "SUCCESS"}
    except:
        print(traceback.format_exc())
        return JSONResponse(content={"message": "FAILURE"}, status_code=500)

@app.post('/scan')
async def sub_(request: Request):
    form = await request.form()
    image = await read_image(request)
    param = [image, 'example', 'test', 'jp']
    keys = ['image', 'foldername', 'filename', 'lang', 'tech']
    for i, key in enumerate(keys[1:]):
        if key in form:
            param[i+1] = form[key]
        else:
            return JSONResponse(content={"message": "MISSING PARAM " + key}, status_code=400)
    
    try:
        texts, bbs = sub(*tuple(param))
        sub_text = ""
        for text, bb in zip(texts, bbs):
            sub_text += bb + '\n' + text
        return {"message": "SUCCESS", "sub_text": sub_text}
    except Exception:
        print(traceback.format_exc())
        return JSONResponse(content={"message": "FAILURE"}, status_code=500)

uvicorn.run(app, host='0.0.0.0', port=8000)