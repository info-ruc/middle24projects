import os
import openai
import json
import cv2
import time
from copy import deepcopy
from math import floor
from all_func import infer


class MovieChatbot(object):
    
    def __init__(self, openai_api_key, google_api, google_cse, prompt_path, video_path, qa_path):
        openai.api_key = openai_api_key
        self.video_path = video_path
        self.qa_path = qa_path
        with open(prompt_path, "r") as f:
            self.prompt = json.load(f)
        self.delay_sec = 4
        self.max_retry_num = 5
        self.model_list = self.prompt["model_list"]
        self.time_model_list = self.prompt["time_model_list"]
        self.transfer_dict = self.prompt["transfer_dict"]
        self.tool_description = self.prompt["tool_description"]
        self.google_api = google_api
        self.google_cse = google_cse
        
        self.movie_title = "夏洛特烦恼" # TODO: support flexible title choosing
        
    def Translation(self, user_input, order):
        time.sleep(self.delay_sec)
        if order == 'ch2en':
            translation_prompt = self.prompt["translation_ch_to_en"]
            model_selection_messages = [
                {"role": "system", "content": translation_prompt},
                {"role": "user", "content": user_input}
            ]
            completion = openai.ChatCompletion.create(model="gpt-3.5-turbo-0613", messages=model_selection_messages)
            result = completion.choices[0].message["content"]
            print("translated: ", result)
        elif order == 'en2ch':
            translation_prompt = self.prompt["translation_en_to_ch"]
            model_selection_messages = [
                {"role": "system", "content": translation_prompt},
                {"role": "user", "content": user_input}
            ]
            completion = openai.ChatCompletion.create(model="gpt-3.5-turbo-0613", messages=model_selection_messages)
            result = completion.choices[0].message["content"]
            print("translated: ", result)
        return result
    
    def DecisionMaking(self, user_input):
        time.sleep(self.delay_sec)
        decision_making_prompt = self.prompt["decision_making"]
        decision_making_messages = deepcopy(user_input)
        decision_making_messages[-1]["content"] = decision_making_prompt + decision_making_messages[-1]["content"]
        # print(decision_making_messages)
        completion = openai.ChatCompletion.create(model="gpt-3.5-turbo-0613", messages=decision_making_messages)
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
        completion = openai.ChatCompletion.create(model="gpt-3.5-turbo-0613", messages=text_inference_messages)
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
        completion = openai.ChatCompletion.create(model="gpt-3.5-turbo-0613", messages=time_parsing_messages)
        info = completion.choices[0].message["content"]
        print("time parsing response: ", info)
        
        l, r = info.find("{"), info.rfind("}")
        if l != -1 and r != -1:
            info = info[l:(r+1)]
            return json.loads(info)["开始"], json.loads(info)["结束"], ""
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

    def ToolPlanning(self, user_input):
        time.sleep(self.delay_sec)
        tool_planning_prompt = self.prompt["tool_planning"] + self.prompt["tool_list"]
        tool_planning_messages = deepcopy(user_input)
        tool_planning_messages[-1]["content"] = tool_planning_prompt + tool_planning_messages[-1]["content"]
        completion = openai.ChatCompletion.create(model="gpt-3.5-turbo-0613", messages=tool_planning_messages)
        info = completion.choices[0].message["content"]
        print("tool planning response: ", info)
        
        l, r = info.find("["), info.rfind("]")
        tool_id = list()
        reason = list()
        step_id = list()
        if l != -1 and r != -1: # json
            try:
                info = json.loads(info[l:(r+1)])
                for model in info:
                    tool_id.append(model["工具名称"])
                    reason.append(model["理由"])
                    step_id.append(model["步骤序号"])
                return tool_id, reason, step_id, ""
            except Exception as e: 
                print(e)
        for model in self.model_list: # other format
            index = info.find(model)
            if index != -1:
                tool_id.append(model)
                reason = info
                for i in range(index, len(info)):
                    step = 0
                    while str(info[i]).isnumeric():
                        step = step*10 + int(info[i])
                        i += 1
                    if step > 0:
                        step_id.append(step)
                        break
                if len(step_id) < len(tool_id):
                    step_id.append(0)
        if len(tool_id) > 0:
            return tool_id, reason, step_id, ""
        return -1, -1, info, ""
        
    def QuestionToQuery(self, question):
        time.sleep(self.delay_sec)
        question2query_prompt = self.prompt["question_to_query"]
        question2query_messages = [{"role": "user", "content": question}]
        question2query_messages.insert(0, {"role": "system", "content": question2query_prompt})
        completion = openai.ChatCompletion.create(model="gpt-3.5-turbo-0613", messages=question2query_messages)
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
        total_input = user_input
        user_input = deepcopy(user_input[-1]["content"])
        results = list()
        for model in selected_model:
            cached_file = ""
            if model in self.time_model_list:
                try:
                    begin, end, info = self.TimeParsing(user_input, self.time_model_list.index(model))
                    if begin == -1 and end == -1:
                        begin, end = user_input[:8], user_input[:8]
                except:
                    begin, end = user_input[:8], user_input[:8]
                begin = int(begin[:2])*3600 + int(begin[3:5])*60 + int(begin[6:])
                end = int(end[:2])*3600 + int(end[3:5])*60 + int(end[6:])
                print("requested time segment: [", begin, end, "]")
                if begin > end:
                    begin = end
                cached_file = self.VideoProcess(begin, end)
            if model == "Image Captioning Model":
                input_src = {
                    "img_url": cached_file,
                    "text": None
                }
            elif model == "Visual Question Answering Model":
                input_src = {
                    "img_url": cached_file,
                    "text": self.Translation(user_input.split(':')[-1], "ch2en")                   # TODO HH:MM:SS: user_request
                }
            elif model == "Video Captioning Model":
                input_src = {
                    "vid_url": cached_file,
                    "text": None
                }
            elif model == "Video Narrating Model":
                input_src = {
                    "movie_id": "6965768652251628068",
                    "starttime": begin,
                    "endtime": end
                }
            elif model == 'Video Grounding Model':
                # user_input = self.Translation(user_input, 'en2ch') # English input
                input_src = {
                    "movie_id": "6965768652251628068",
                    "query": self.QuestionToQuery(user_input.split(':')[-1])
                }
            elif model == 'Actor or Role Recognition':
                input_src = {
                    "movie_id": "6965768652251628068",
                    "begin": begin,
                    "end": end,
                    "user_input": user_input.split(':')[-1] 
                }
            elif model == 'Movie Information':
                input_src = {
                    "movie_id": "6965768652251628068"
                }
            if model == 'Search Engine':
                result = self.SearchTool(total_input, results)
            else:
                result = infer(self.transfer_dict[model], input_src)
            
            if model in  ["Image Captioning Model", "Visual Question Answering Model", "Video Captioning Model"]: # English
                result = self.Translation(str(result), 'en2ch')
            if model == "Video Grounding Model":
                start, end = result[0], result[1]
                start_t = str(floor(start/3600)) + ":" + str(floor(start/60)) + ":" + str(floor(start%60))
                end_t = str(floor(end/3600)) + ":" + str(floor(end/60)) + ":" + str(floor(end%60))
                result = [start_t, end_t]
            results.append({"结果描述": self.tool_description[model], "返回结果": result})
            if cached_file != "":
                os.remove(cached_file)
        return results
    
    def Search(self, user_input):
        time.sleep(self.delay_sec)
        
        # google
        input_src = {
            "query": user_input[9:] + " " + self.movie_title, #TODO HH:MM:SS
            "num_results": 20,
            "api_key": self.google_api,
            "cse_id": self.google_cse
        }
        search_result = str(infer("google-search", input_src))
        
        search_prompt = self.prompt["search_result_summarization"]
        search_prompt = search_prompt.replace("User_Input", user_input)
        search_prompt = search_prompt.replace("Search_Result", search_result)
        search_message = [{"role": "user", "content": search_prompt}]
        completion = openai.ChatCompletion.create(model="gpt-3.5-turbo-0613", messages=search_message)
        search_result = completion.choices[0].message["content"]
        print("search summarization result: ", search_result)
        
        time.sleep(self.delay_sec)
        
        # qa
        with open("info/qa/夏洛特烦恼.json") as f:
            qa_list = json.load(f)
        
        questions = list()
        qa_list = qa_list[list(qa_list.keys())[0]]["QA_list"]
        for i in range(len(qa_list)):
            q = qa_list[i]
            questions.append({"问题序号": i+1, "问题内容": q["question"]})
        retrieval_prompt = self.prompt["question_retrieval"]
        retrieval_prompt = retrieval_prompt.replace("Questions", str(questions))
        retrieval_prompt = retrieval_prompt.replace("User_Input", user_input[9:])
        retrieval_message = [{"role": "user", "content":  retrieval_prompt}]
        completion = openai.ChatCompletion.create(model="gpt-3.5-turbo-0613", messages=retrieval_message)
        qa_result = completion.choices[0].message["content"]
        for i in range(len(qa_result)):
            q_idx = 0
            while i != len(qa_result) and str(qa_result[i]).isnumeric():
                q_idx = q_idx*10 + int(qa_result[i])
                i += 1
            if q_idx > 0:
                q_idx -= 1
                ans_idx = qa_list[q_idx]["correct_index"]
                qa_result = qa_list[q_idx]["question"] + qa_list[q_idx]["answers"][ans_idx] + "证据：" + str(qa_list[q_idx]["CoT"])
                break
        print("qa retrieval result: ", qa_result)
        
        # merge
        merge_prompt = self.prompt["merge_search_retrieval_result"]
        merge_prompt = merge_prompt.replace("User_Input", user_input[9:])
        merge_prompt = merge_prompt.replace("Search_Result", search_result)
        merge_prompt = merge_prompt.replace("QA_result", qa_result)
        merge_message = [{"role": "user", "content":  merge_prompt}]
        completion = openai.ChatCompletion.create(model="gpt-3.5-turbo-0613", messages=merge_message)
        result = completion.choices[0].message["content"]
        print("merged search result: ", result)
        
        return result
    
    def SearchTool(self, total_input, results):
        time.sleep(self.delay_sec)
        search_prompt = self.prompt["search_content_decision"]
        search_prompt = search_prompt.replace("Result", str(results))
        search_prompt = search_prompt.replace("User_Input", total_input[-1]["content"][9:])
        search_message = deepcopy(total_input[1:-1])
        search_message.append({"role": "user", "content": search_prompt})
        completion = openai.ChatCompletion.create(model="gpt-3.5-turbo-0613", messages=search_message)
        result = completion.choices[0].message["content"]
        print("search tool result: ", result)
        
        input_src = {
            "query": result, #TODO HH:MM:SS
            "num_results": 20,
            "api_key": self.google_api,
            "cse_id": self.google_cse
        }
        search_result = str(infer("google-search", input_src))
        
        time.sleep(self.delay_sec)
        search_prompt = self.prompt["search_result_summarization"]
        search_prompt = search_prompt.replace("User_Input", result)
        search_prompt = search_prompt.replace("Search_Result", search_result)
        search_message = [{"role": "user", "content": search_prompt}]
        completion = openai.ChatCompletion.create(model="gpt-3.5-turbo-0613", messages=search_message)
        search_result = completion.choices[0].message["content"]
        print("search summarization result: ", search_result)
        
        return search_result
        
    def AnswerSummarization(self, user_input, result, search_result):
        time.sleep(self.delay_sec)
        user_request, requested_time = user_input[-1]['content'].split(":")[-1], user_input[-1]['content'].split(":")[:-1] 
        answer_summarization_prompt = self.prompt["answer_summarization"]
        result.append({"结果描述": self.tool_description["Search Result"], "结果": search_result})
        
        answer_summarization_prompt = answer_summarization_prompt.replace("User_Input", user_request)
        answer_summarization_prompt = answer_summarization_prompt.replace("Result", str(result))
        answer_summarization_messages = user_input[:-1]
        answer_summarization_messages.append({"role": "user", "content": answer_summarization_prompt})
        # print(answer_summarization_messages)
        completion = openai.ChatCompletion.create(model="gpt-3.5-turbo-0613", messages=answer_summarization_messages)
        info = completion.choices[0].message["content"]
        print("answer summarization response: ", info)
        return info
    
    def JudgeAnswer(self, answer, user_input):
        time.sleep(self.delay_sec)
        judge_answer_prompt = self.prompt["judge_answer"]
        judge_answer_prompt = judge_answer_prompt.replace("User_Input", user_input[-1]["content"][9:])
        judge_answer_prompt = judge_answer_prompt.replace("Response", answer)
        judge_answer_messages = user_input[:-1] + [{"role": "user", "content": judge_answer_prompt}]
        completion = openai.ChatCompletion.create(model="gpt-3.5-turbo-0613", messages=judge_answer_messages)
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
        search_result = self.Search(user_input[-1]["content"])
        while retry_time < self.max_retry_num:
            flag = self.DecisionMaking(user_input)
            if flag == True:
                selected_model, reason, step_id, info = self.ToolPlanning(user_input)
                if selected_model == -1 and reason == -1:
                    return info
                print("selected model: ", selected_model)
                print("reason: ", reason)
                print("order: ", step_id)
                
                selected_model, reason = self.StepOrder(selected_model, reason, step_id)
                result = self.Inference(selected_model, user_input)
                print("inference result: ", result)
                answer = self.AnswerSummarization(user_input, result, search_result)
            else:
                answer = self.TextInference(user_input)
            if self.JudgeAnswer(answer, user_input):
                break
            retry_time += 1
        return answer
        