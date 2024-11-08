import argparse
import logging
import random
import jsonlines
import uuid
import numpy as np
from diffusers.utils import load_image
from transformers import pipeline
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan, SpeechT5ForSpeechToSpeech
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from datasets import load_dataset
from PIL import Image
import flask
from flask import request, jsonify
import waitress
from flask_cors import CORS
import io
from torchvision import transforms
import torch
import torchaudio
from huggingface_hub import hf_hub_url, cached_download
from transformers import AutoImageProcessor, TimesformerForVideoClassification
from transformers import MaskFormerFeatureExtractor, MaskFormerForInstanceSegmentation, AutoFeatureExtractor
from transformers import DPTForDepthEstimation, DPTFeatureExtractor
import warnings
import time
import traceback
import os
import yaml
from model_infer.SwinBERT import VideoCaptioning
from model_infer.video_narrating import VideoNarrating
from model_infer.grounding import grounding_infer
from model_infer.movie_info import actor_info, movie_intro
from model_infer.search import search_google
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
# parser.add_argument("--config", type=str, default="configs/config.default.yaml")
args = parser.parse_args()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)

# host = config["local_inference_endpoint"]["host"]
# port = config["local_inference_endpoint"]["port"]

# local_deployment = config["local_deployment"]
device = 0

# PROXY = None
# if config["proxy"]:
#     PROXY = {
#         "https": config["proxy"],
#     }

app = flask.Flask(__name__)
CORS(app)

start = time.time()

local_fold = "models"
# if args.config.endswith(".dev"):
#     local_fold = "models_dev"

pipeline_models = ["Salesforce/blip-image-captioning-base", "facebook/detr-resnet-101", "dandelin/vilt-b32-finetuned-vqa"]

def load_pipes():

    standard_pipes = {
        "Salesforce/blip-image-captioning-base": {
            "model": pipeline(task="image-to-text", model=f"Salesforce/blip-image-captioning-base"), 
            "processor": BlipProcessor.from_pretrained(f"Salesforce/blip-image-captioning-large"),
            "device": device
        },
        # "facebook/detr-resnet-101": {
        #     "model": pipeline(task="object-detection", model=f"facebook/detr-resnet-101"), 
        #     "device": device
        # },
        "dandelin/vilt-b32-finetuned-vqa": {
            "model": pipeline(task="visual-question-answering", model=f"dandelin/vilt-b32-finetuned-vqa"), 
            "device": device
        }
    }

    pipes = {**standard_pipes}
    return pipes

pipes = load_pipes()

end = time.time()
during = end - start

print(f"[ ready ] {during}s")


def infer(model_id, input_src):

    start = time.time()
    
    if model_id in pipeline_models:
        pipe = pipes[model_id]["model"]
        
        if "device" in pipes[model_id]:
            try:
                pipe.to(pipes[model_id]["device"])
            except:
                pipe.device = torch.device(pipes[model_id]["device"])
                pipe.model.to(pipes[model_id]["device"])
    
    result = None
    try:
        # image to text
        if model_id == "Salesforce/blip-image-captioning-base":
            # raw_image = load_image(input_src["img_url"]).convert('RGB')
            # inputs = pipes[model_id]["processor"](raw_image, return_tensors="pt").to(pipes[model_id]["device"])
            # out = pipe.generate(**inputs)
            result = pipe(images=input_src["img_url"])
            # caption = pipes[model_id]["processor"].decode(out[0], skip_special_tokens=True)
        
        # video to text
        if model_id == "SwinBERT-video-captioning":
            result = VideoCaptioning(video=input_src["vid_url"])
        
        # VQA
        if model_id == "dandelin/vilt-b32-finetuned-vqa":
            question = input_src["text"]
            img_url = input_src["img_url"]
            result = pipe(question=question, image=img_url)
            
        # Narrating
        if model_id == "video-narrating":
            
            movie_id = input_src["movie_id"]
            starttime = input_src["starttime"]
            endtime = input_src["endtime"]
            result = VideoNarrating(movie_id=movie_id, starttime=starttime, endtime=endtime)
            
        # Grounding
        if model_id == "video-grounding":
            movie_id = input_src["movie_id"]
            query = input_src["query"]
            result = grounding_infer(movie_id=movie_id, query=query)
            
        # Movie infomation
        if model_id == "actor-info":
            movie_id = input_src["movie_id"]
            begin = input_src["begin"]
            end = input_src["end"]
            user_input = input_src["user_input"]
            result = actor_info(movie_id=movie_id, begin=begin, end=end, user_input=user_input)
        
        if model_id == "movie-intro":
            movie_id = input_src["movie_id"]
            result = movie_intro(movie_id=movie_id)
        
        # Google search
        if model_id == "google-search":
            query = input_src["query"]
            num_results = input_src["num_results"]
            api_key = input_src["api_key"]
            cse_id = input_src["cse_id"]
            result = search_google(query=query, num_results=num_results, api_key=api_key, cse_id=cse_id)
                    
        
    except Exception as e:
        print(e)
        traceback.print_exc()
        result = {"error": {"message": "Error when running the model inference."}}
        
    if model_id in pipeline_models:
        if "device" in pipes[model_id]:
            try:
                pipe.to("cpu")
                torch.cuda.empty_cache()
            except:
                pipe.device = torch.device("cpu")
                pipe.model.to("cpu")
                torch.cuda.empty_cache()

        pipes[model_id]["using"] = False

    if result is None:
        result = {"error": {"message": "model not found"}}
    
    end = time.time()
    during = end - start
    print(f"[ complete {model_id} ] {during}s")
    print(f"[ result {model_id} ] {result}")

    # return jsonify(result)
    return result

if __name__ == '__main__':

    model_id = "Salesforce/blip-image-captioning-base"
    input_src = {
        "img_url": "/data5/yzh/ChatMovie/models/ohyeah.png",
        "text": None
    }
    infer(model_id, input_src)

    # model_id = "dandelin/vilt-b32-finetuned-vqa"
    # input_src = {
    #     "img_url": "/data5/yzh/ChatMovie/models/ohyeah.png",
    #     "text": "what is the animal?"
    # }
    # infer(model_id, input_src)
    
    # model_id = "SwinBERT-video-captioning"
    # input_src = {
    #     "vid_url": "/data4/myt/SwinBERT/docs/G0mjFqytJt4_000152_000162.mp4"
    # }
    # infer(model_id, input_src)
    
    
    # model_id = "video-narrating"
    # input_src = {
    #     "movie_id": "6965768652251628068",
    #     "starttime": 1501 ,
    #     "endtime": 1509
    # }
    # infer(model_id, input_src)
    
    # with jsonlines.open('/data4/myt/MovieChat/test/narrating-1.jsonl', mode='a') as writer:
    #     for starttime in range(2400, 6000, 5):
    #         model_id = "video-narrating"
    #         input_src = {
    #             "movie_id": "6965768652251628068",
    #             "starttime": starttime ,
    #             "endtime": starttime+5
    #         }
    #         result = infer(model_id, input_src)
    #         writer.write({'start': str(starttime//60)+":"+str(starttime%60), 'end': 5, 'result': result})
    #         model_id = "video-narrating"
    #         input_src = {
    #             "movie_id": "6965768652251628068",
    #             "starttime": starttime ,
    #             "endtime": starttime+10
    #         }
    #         result = infer(model_id, input_src)
    #         writer.write({'start': str(starttime//60)+":"+str(starttime%60), 'end': 10, 'result': result})
    
    # model_id = "video-grounding"
    # input_src = {
    #     "movie_id": "6965768652251628068",
    #     "query": "夏洛正在弹吉他"
    # }
    # infer(model_id, input_src)
    
    # model_id = "movie-intro"
    # input_src = {
    #     "movie_id": "6965768652251628068"
    # }
    # infer(model_id, input_src)
    
    # model_id = "google-search"
    # input_src = {
    #     "query": "夏洛为什么觉得自己一点尊严也没有了。",
    #     "num_results": 20,
    #     "api_key": "AIzaSyDBmaMvqXWonpLBATmaFJ4USQJIuVaxajY",
    #     "cse_id": "808b72bcb7c994eea"
    # }
    # infer(model_id, input_src)