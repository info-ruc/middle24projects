import os
import openai
import json
import cv2
import time
from copy import deepcopy

from all_func import infer

class MovieChatbot(object):
    
    def __init__(self, openai_api_key, prompt_path, video_path):
        openai.api_key = openai_api_key
        self.video_path = video_path
        with open(prompt_path, "r") as f:
            self.prompt = json.load(f)
        self.delay_sec = 2
        self.max_retry_num = 5
        self.model_list = self.prompt["model_list"]
        self.time_model_list = self.prompt["time_model_list"]
        
    def Translation(self, user_input, order):
        time.sleep(self.delay_sec)
        if order == 'ch2en':
            translation_prompt = self.prompt["translation_ch_to_en"]
            model_selection_messages = [
                {"role": "system", "content": translation_prompt},
                {"role": "user", "content": user_input}
            ]
            completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=model_selection_messages)
            result = completion.choices[0].message["content"]
            print("translated: ", result)
        elif order == 'en2ch':
            translation_prompt = self.prompt["translation_en_to_ch"]
            model_selection_messages = [
                {"role": "system", "content": translation_prompt},
                {"role": "user", "content": user_input}
            ]
            completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=model_selection_messages)
            result = completion.choices[0].message["content"]
            print("translated: ", result)
        return result
    
    def DecisionMaking(self, user_input):
        time.sleep(self.delay_sec)
        decision_making_prompt = self.prompt["decision_making"]
        decision_making_messages = deepcopy(user_input)
        decision_making_messages[-1]["content"] = decision_making_prompt + decision_making_messages[-1]["content"]
        # print(decision_making_messages)
        completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=decision_making_messages)
        info = completion.choices[0].message["content"]
        print("decision making response: ", info)
        if info.find("yes") != -1 or info.find("Yes") != -1:
            return True
        else:
            return False
        
    def TextInference(self, user_input):
        time.sleep(self.delay_sec)
        text_inference_prompt = self.prompt["text_inference"]
        text_inference_messages = deepcopy(user_input)
        text_inference_messages[-1]["content"] = text_inference_prompt + text_inference_messages[-1]["content"]
        # print(text_inference_messages)
        completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=text_inference_messages)
        info = completion.choices[0].message["content"]
        print("text inference response: ", info)
        return info
        
    def TimeParsing(self, user_input, model_index):
        time.sleep(self.delay_sec)
        time_parsing_prompt = self.prompt["time_parsing"]
        time_parsing_prompt = time_parsing_prompt.replace("Selected_Modle", self.time_model_list[model_index])
        time_parsing_prompt = time_parsing_prompt.replace("Model_Description", self.prompt["time_model_description"][model_index])
        time_parsing_messages = user_input
        time_parsing_messages= [{"role": "user", "content": time_parsing_prompt + time_parsing_messages}]
        completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=time_parsing_messages)
        info = completion.choices[0].message["content"]
        print("time parsing response: ", info)
        
        l, r = info.find("{"), info.rfind("}")
        if l != -1 and r != -1:
            info = info[l:(r+1)]
            return json.loads(info)["begin"], json.loads(info)["end"], ""
        else:
            idx = info.find(":")
            if idx != -1 and info[idx+3] == ":": # HH:MM:SS
                return  info[idx-2:idx+5], info[idx-2:idx+5], ""
            return -1, -1, info

    def VideoProcess(self, begin, end):
        cap = cv2.VideoCapture(self.video_path)
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.set(cv2.CAP_PROP_POS_FRAMES, begin * fps)
        pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        if begin == end: # image
            _, frame = cap.read()
            file_path = "/data4/myt/MovieChat/cache/{}_{}.png".format(begin, end)
            cv2.imwrite(file_path, frame)
            print("cached file path: ", file_path)
            cap.release()
            return file_path
        else: # video
            file_path = "/data4/myt/MovieChat/cache/{}_{}.mp4".format(begin, end)
            out = cv2.VideoWriter(file_path, fourcc, fps, (int(width), int(height)))
            while (pos <= end * fps):
                _, frame = cap.read()
                out.write(frame)
                pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
            print("cached file path: ", file_path)
            cap.release()
            out.release()
            return file_path

    def ModelSelection(self, user_input):
        time.sleep(self.delay_sec)
        model_seletion_prompt = self.prompt["model_selection"] + self.prompt["model_list_info"]
        model_selection_messages = deepcopy(user_input)
        model_selection_messages[-1]["content"] = model_seletion_prompt + model_selection_messages[-1]["content"]
        completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=model_selection_messages)
        info = completion.choices[0].message["content"]
        print("model selection response: ", info)
        
        l, r = info.find("["), info.rfind("]")
        model_id = list()
        reason = list()
        step_id = list()
        if l != -1 and r != -1: # json
            try:
                info = json.loads(info[l:(r+1)])
                for model in info:
                    model_id.append(model["model_id"])
                    reason.append(model["reason"])
                    step_id.append(model["step_id"])
                return model_id, reason, step_id, ""
            except Exception as e: 
                print(e)
        for model in self.model_list: # other format
            index = info.find(model)
            if index != -1:
                model_id.append(model)
                reason = info
                for i in range(index, len(info)):
                    step = 0
                    while str(info[i]).isnumeric():
                        step = step*10 + int(info[i])
                    if step > 0:
                        step_id.append(step)
                        break
                if len(step_id) < len(model_id):
                    step_id.append(0)
        if len(model_id) > 0:
            return model_id, reason, step_id, ""
        return -1, -1, info, ""
        
    def QuestionToQuery(self, question):
        time.sleep(self.delay_sec)
        question2query_prompt = self.prompt["question_to_query"]
        question2query_messages = [{"role": "user", "content": question}]
        question2query_messages.insert(-1, {"role": "system", "content": question2query_prompt})
        completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=question2query_messages)
        info = completion.choices[0].message["content"]
        print("q2q response: ", info)
        return info

    def StepOrder(self, selected_model, reason, step_id):
        ordered_model = list()
        ordered_reason = list()
        step_index = list()
        for i in range(len(step_id)):
            step_index.append([step_id[i], i])
        step_index.sort(key=lambda x:x[0])
        for sid_idx in step_index:
            ordered_model.append(selected_model[sid_idx[1]])
            ordered_reason.append(reason[sid_idx[1]])
        return ordered_model, ordered_reason

    def Inference(self, selected_model, user_input):
        user_input = deepcopy(user_input[-1]["content"])
        results = list()
        for model in selected_model:
            cached_file = ""
            if model in self.time_model_list:
                try:
                    begin, end, info = self.TimeParsing(user_input, self.time_model_list.index(model))
                except:
                    begin, end = user_input[:8], user_input[:8]
                if begin == -1 and end == -1:
                    begin, end = user_input[:8], user_input[:8]
                begin = int(begin[:2])*3600 + int(begin[3:5])*60 + int(begin[6:])
                end = int(end[:2])*3600 + int(end[3:5])*60 + int(end[6:])
                print("requested time segment: [", begin, end, "]")
                if begin > end:
                    return "Can't identify requested time."
                cached_file = self.VideoProcess(begin, end)
            if model == "Salesforce/blip-image-captioning-base":
                input_src = {
                    "img_url": cached_file,
                    "text": None
                }
            elif model == "dandelin/vilt-b32-finetuned-vqa":
                input_src = {
                    "img_url": cached_file,
                    "text": user_input.split(':')[-1]                   # TODO HH:MM:SS: user_request
                }
            elif model == "SwinBERT-video-captioning":
                input_src = {
                    "vid_url": cached_file,
                    "text": None
                }
            elif model == "video-narrating":
                input_src = {
                    "movie_id": "6965768652251628068",
                    "starttime": begin,
                    "endtime": end
                }
            elif model == 'video-grounding':
                user_input = self.QuestionToQuery(user_input)
                user_input = self.Translation(user_input, 'en2ch')
                input_src = {
                    "movie_id": "6965768652251628068",
                    "query": user_input
                }
            elif model == 'actor-info':
                input_src = {
                    "movie_id": "6965768652251628068",
                    "begin": begin,
                    "end": end,
                    "user_input": user_input.split(':')[-1] 
                }
            elif model == 'movie-intro':
                input_src = {
                    "movie_id": "6965768652251628068"
                }
            result = infer(model, input_src)
            
            if model in  ["video-narrating", "movie-intro", "actor-info"]: # Chinese
                result = self.Translation(str(result), 'ch2en')
            results.append([model, result])
            if cached_file != "":
                os.remove(cached_file)
        return results
    
    def AnswerSummarization_long(self, user_input, result, selected_model, reason):
        time.sleep(self.delay_sec)
        user_request, requested_time = user_input[-1]['content'].split(":")[-1], user_input[-1]['content'].split(":")[:-1] 
        answer_summarization_prompt = self.prompt["answer_summarization_long"]
            
        answer_summarization_prompt = answer_summarization_prompt.replace("User_Input", user_request)
        answer_summarization_prompt = answer_summarization_prompt.replace("Requested_Time", str(requested_time))
        answer_summarization_prompt = answer_summarization_prompt.replace("Model_Selection", selected_model)
        answer_summarization_prompt = answer_summarization_prompt.replace("Selecting_Reason", reason)
        answer_summarization_prompt = answer_summarization_prompt.replace("Inference_Result", result)
        answer_summarization_messages = user_input[:-1]
        answer_summarization_messages.append({"role": "user", "content": answer_summarization_prompt})
        # print(answer_summarization_messages)
        completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=answer_summarization_messages)
        info = completion.choices[0].message["content"]
        print("answer summarization response: ", info)
        return info
    
    def AnswerSummarization(self, user_input, result, selected_model, reason):
        time.sleep(self.delay_sec)
        user_request, requested_time = user_input[-1]['content'].split(":")[-1], user_input[-1]['content'].split(":")[:-1] 
        answer_summarization_prompt = self.prompt["answer_summarization_short"]
            
        answer_summarization_prompt = answer_summarization_prompt.replace("User_Input", user_request)
        answer_summarization_prompt = answer_summarization_prompt.replace("Inference_Result", result)
        answer_summarization_messages = user_input[:-1]
        answer_summarization_messages.append({"role": "user", "content": answer_summarization_prompt})
        # print(answer_summarization_messages)
        completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=answer_summarization_messages)
        info = completion.choices[0].message["content"]
        print("answer summarization response: ", info)
        return info
    
    def JudgeAnswer(self, answer, user_input):
        time.sleep(self.delay_sec)
        judge_answer_prompt = self.prompt["judge_answer"]
        judge_answer_messages = [
            {"role": "system", "content": judge_answer_prompt},
            {"role": "user", "content": "user's request: " + user_input[-1]["content"]},
            {"role": "user", "content": "response: " + answer}
        ]
        print(judge_answer_messages)
        completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=judge_answer_messages)
        result = completion.choices[0].message["content"]
        print("judge answer: ", result)
        if result.find("yes") != -1 or result.find("Yes") != -1:
            return True
        else:
            return False
        
    def UserRequest(self, user_input):  
        while len(user_input) >= 10:
            user_input.pop(0)
        user_input = deepcopy(user_input)
        user_input.insert(0,  {"role": "system", "content": self.prompt["system_prompt"]})
        answer = ""
        retry_time = 0
        while retry_time < self.max_retry_num:
            flag = self.DecisionMaking(user_input)
            if flag == True:
                selected_model, reason, step_id, info = self.ModelSelection(user_input)
                if selected_model == -1 and reason == -1:
                    return info
                print("selected model: ", selected_model)
                print("reason: ", reason)
                print("order: ", step_id)
                
                selected_model, reason = self.StepOrder(selected_model, reason, step_id)
                result = self.Inference(selected_model, user_input)
                print("inference result: ", result)
                answer = self.AnswerSummarization(user_input, str(result), str(selected_model), str(reason))
            else:
                answer = self.TextInference(user_input)
            if self.JudgeAnswer(answer, user_input):
                break
            retry_time += 1
        return answer
        