import os
import json
import tqdm
import sys
sys.path.append('/data5/yzh/MovieUN_v2/MovieUN-G/data_process')
sys.path.append('/data5/yzh/MovieUN_v2/CNCLIP')
from build_movie_itm import build_movie_data
from cn_clip.preprocess.tojsonl import query_tojsonl
from cn_clip.preprocess.tobase64 import tobase64
import pdb

resume_path = '/data5/yzh/MovieUN_v2/CNCLIP/experiments/epoch_1_500.pt'

# build movie data for feature extraction
def preprocess(movie_id, query):
    cached_movies = os.listdir('/data5/yzh/MovieUN_v2/CNCLIP/datasets/movieun/movie_data')
    json_data_path = build_movie_data(movie_id)
    if movie_id not in cached_movies:
        jsonl_data_path = query_tojsonl(query, movie_id)
        tsv_data_path = tobase64(json_data_path, movie_id)
        os.system(f'python /data5/yzh/MovieUN_v2/CNCLIP/cn_clip/preprocess/build_lmdb_dataset.py --data_dir /data5/yzh/MovieUN_v2/CNCLIP/datasets/movieun/movie_data/{movie_id} --splits test')

# extract features
def extract_features(movie_id, query):
    preprocess(movie_id, query)
    image_feats_path = f'/data5/yzh/MovieUN_v2/CNCLIP/datasets/movieun/movie_data/{movie_id}/test_img_feats.npy'
    text_feats_path = f'/data5/yzh/MovieUN_v2/CNCLIP/datasets/movieun/movie_data/{movie_id}/test_text_feats.npy'
    os.system(f'export CUDA_VISIBLE_DEVICES=7 && export PYTHONPATH=/data5/yzh/MovieUN_v2/CNCLIP && python -u /data5/yzh/MovieUN_v2/CNCLIP/cn_clip/eval/extract_features.py \
                --extract-text-feats \
                --text-data="/data5/yzh/MovieUN_v2/CNCLIP/datasets/movieun/movie_data/{movie_id}/test_texts.jsonl" \
                --text-feat-output-path="{text_feats_path}" \
                --text-batch-size=32 \
                --context-length=200 \
                --resume="{resume_path}" \
                --vision-model=ViT-H-14 \
                --text-model=RoBERTa-wwm-ext-large-chinese')
    cached_features = os.listdir(f'/data5/yzh/MovieUN_v2/CNCLIP/datasets/movieun/movie_data/{movie_id}')
    if 'test_img_feats.npy' not in cached_features:
        os.system(f'export CUDA_VISIBLE_DEVICES=7 && export PYTHONPATH=/data5/yzh/MovieUN_v2/CNCLIP && python -u /data5/yzh/MovieUN_v2/CNCLIP/cn_clip/eval/extract_features.py \
                    --extract-image-feats \
                    --image-data="/data5/yzh/MovieUN_v2/CNCLIP/datasets/movieun/movie_data/{movie_id}/lmdb/test/imgs" \
                    --image-feat-output-path="{image_feats_path}" \
                    --img-batch-size=32 \
                    --context-length=200 \
                    --resume="{resume_path}" \
                    --vision-model=ViT-H-14 \
                    --text-model=RoBERTa-wwm-ext-large-chinese')
    else:
        print('USE CACHED IMAGE FEATURE!!')
    
    return image_feats_path, text_feats_path

# make predictions
def retrieve(movie_id, query):
    image_feats_path, text_feats_path = extract_features(movie_id, query)
    os.system(f'python -u /data5/yzh/MovieUN_v2/CNCLIP/cn_clip/eval/make_topk_predictions.py \
    --image-feats="{image_feats_path}" \
    --text-feats="{text_feats_path}" \
    --top-k=1 \
    --eval-batch-size=32768 \
    --output="/data5/yzh/MovieUN_v2/CNCLIP/datasets/movieun/movie_data/{movie_id}/retrieval_res.jsonl"')

    # read res
    with open(f'/data5/yzh/MovieUN_v2/CNCLIP/datasets/movieun/movie_data/{movie_id}/retrieval_res.jsonl', 'r') as f:
        res = json.load(f)
    video_id = res['image_ids'][0] # video_id in /data5/yzh/MovieUN_v2/narrations/video2text_meta.json
    # get start_time and end_time
    with open('/data5/yzh/MovieUN_v2/narrations/video2text_meta.json', 'r') as f:
        video2text = json.load(f)
        video_info = video2text[movie_id][str(video_id)]
        start = video_info['start']
        end = video_info['end']

    return start, end

from transformers import AutoTokenizer, AutoModel
class ChineseBert(object):
    def __init__(self):
       self.tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
       self.model = AutoModel.from_pretrained("bert-base-chinese")
    
    def __call__(self,text):
        encoded_input = self.tokenizer(text,return_tensors="pt")
        query_token = self.tokenizer(text, padding="max_length",truncation=True, max_length=4)
        word_lens = query_token['attention_mask']
        words_vec = query_token['input_ids']
        output = self.model(**encoded_input)
        last_hidden_state, pooler_output = output[0], output[1]
        return last_hidden_state, pooler_output
    
def grounding(movie_id, anchor_start, anchor_end, query):

    # decide extended time span
    pred_center = (anchor_start + anchor_end) / 2

    import numpy as np
    s3d_movie = np.load(os.path.join('/data5/yzh/DATASETS/Movie101/feature/s3d', '%s-4.npz' % movie_id))['features'].astype(np.float32)
    movie_end = s3d_movie.shape[0]
    movie_start = 0

    clip_start = int(max(movie_start, pred_center - 100))
    clip_end = int(min(movie_end, pred_center + 100))
    clip_length = clip_end - clip_start

    # load video features

    face_movie = np.load(os.path.join('/data5/yzh/DATASETS/Movie101/feature/frame_face', '%s.mp4.npy.npz' % movie_id))['features'].astype(np.float32)
    s3d_feature = s3d_movie[clip_start:clip_end]
    face_feature = face_movie[clip_start:clip_end]
    # 将clip和s3d的特征拼接
    video_feature = np.concatenate((s3d_feature, face_feature), axis=1)

    # load text features
    bert = ChineseBert()
    word_vec, feat = bert(query)
    text_feature = word_vec[0]

    # extract names from query
    def extract_rolename(query):
        from transformers import BertTokenizer
        tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        meta_path = f'/data5/yzh/DATASETS/MovieUN/metadata/movie_intro/{movie_id}.json'
        meta = json.load(open(meta_path, 'r', encoding='utf-8'))
        movie_roles = []
        for role in meta['actorList']:
            if 'roleName' in role:
                movie_roles.append([role['roleName'], role['celebrityId']])
        role_token = [] # 记录 narration 中的第几个 token 是 role name，以及对应的 role id
        tokenized_text = tokenizer.tokenize(query)
        new_narration = ''
        for token in tokenized_text:
            new_narration += token[0]
        for role in movie_roles:
            if role[0] in new_narration:
                start = 0
                while True:
                    start = new_narration.find(role[0], start)
                    if start == -1:
                        break
                    role_token += [[start+i, role[1]] for i in range(len(role[0]))]
                    start += len(role[0])
        return role_token
    role_token_list = extract_rolename(query)
    role_list = [role_token[1] for role_token in role_token_list]
    role_list = list(set(role_list))

    # load portrait features
    import h5py
    face_profile_h5 =  h5py.File('/data5/yzh/DATASETS/MovieUN/metadata/face_profile_512dim.hdf5', "r")
    face_feats = []
    for portrait_id in role_list:
        if portrait_id in face_profile_h5:
            face_feats.append(face_profile_h5[portrait_id]['features'])
    is_empty = 0
    if len(face_feats) > 0:
        face_feats = np.array(face_feats).astype(np.float32)
    else:
        is_empty = 1
        face_feats = np.zeros((1,512)).astype(np.float32)
    
    sys.path.append('/data5/yzh/MovieUN_v2/IANET/code')
    import criteria

    from models_.gcn_final import Model
    from utils import generate_anchors

    id2pos = []
    adj_mat = []
    adj_mat = np.asarray(adj_mat)
    start_frame = 0
    end_frame = clip_length
    label = np.asarray([start_frame, end_frame]).astype(np.int32)
    id2pos = np.asarray(id2pos).astype(np.int64)

    max_num_frames = 200
    anchors = generate_anchors('ActivityNet')
    widths = anchors[:, 1] - anchors[:, 0] + 1
    centers = np.arange(0, max_num_frames)
    start_offsets = np.expand_dims(centers, 1) - 0.5 * (np.expand_dims(widths, 0) - 1)
    end_offsets = np.expand_dims(centers, 1) + 0.5 * (np.expand_dims(widths, 0) - 1)

    proposals = np.stack([start_offsets, end_offsets], -1)
    proposals = np.reshape(proposals, [-1, 2])
    illegal = np.logical_or(proposals[:, 0] < 0, proposals[:, 1] >= clip_length)
    label_mask = (1 - illegal).astype(np.uint8)
    label1 = np.repeat(np.expand_dims(label, 0), proposals.shape[0], 0)
    IoUs = criteria.calculate_IoU_batch((proposals[:, 0], proposals[:, 1]), (label1[:, 0], label1[:, 1]))
    IoUs[illegal] = 0.0
    max_IoU = np.max(IoUs)
    IoUs = IoUs / max_IoU
    scores = IoUs.astype(np.float32)
    scores_mask = (1 - illegal).astype(np.uint8)

    max_num_words = 80
    words = text_feature.cpu().detach().numpy()
    ori_words_len = words.shape[0]
    # word padding
    if ori_words_len < max_num_words:
        word_mask = np.zeros([max_num_words], np.uint8)
        word_mask[range(ori_words_len)] = 1
        words = np.pad(words, ((0, max_num_words - ori_words_len), (0, 0)), mode='constant')
    else:
        word_mask = np.ones([max_num_words], np.uint8)
        words = words[:max_num_words]

    # face padding
    max_num_faces = 3
    ori_face_len = face_feats.shape[0]
    # word padding
    if ori_face_len < max_num_faces:
        face_mask = np.zeros([max_num_faces], np.uint8)
        if not is_empty:
            face_mask[range(ori_face_len)] = 1
        face_feats = np.pad(face_feats, ((0, max_num_faces - ori_face_len), (0, 0)), mode='constant')
    else:
        face_mask = np.ones([max_num_faces], np.uint8)
        face_feats = face_feats[:max_num_faces]

    import torch
    
    words = torch.from_numpy(words).unsqueeze(0).cuda()
    word_mask = torch.from_numpy(word_mask).unsqueeze(0).cuda()

    model_input = {
        'frames': torch.from_numpy(video_feature).unsqueeze(0).cuda(),
        'frame_mask': torch.from_numpy(np.ones([clip_length], np.uint8)).unsqueeze(0).cuda(),
        'words': words,
        'word_mask': word_mask,
        'label': torch.from_numpy(scores).unsqueeze(0).cuda(),
        'label_mask': torch.from_numpy(scores_mask).unsqueeze(0).cuda(),
        'gt': torch.from_numpy(label).unsqueeze(0).cuda(),
        'node_pos': torch.from_numpy(id2pos).unsqueeze(0).cuda(),
        'node_mask': word_mask,
        'adj_mat': None,
        'face_feats': torch.from_numpy(face_feats).unsqueeze(0).cuda(),
        'face_mask': torch.from_numpy(face_mask).unsqueeze(0).cuda(),
    }

    model_args = {
        'train': False,
        'evaluate': True,
        'dataset': 'ActivityNet',
        # 'train_data': '/data5/yzh/MovieUN_v2/MovieUN-G/grounding/train_role.json',
        # 'val_data': '/data5/yzh/MovieUN_v2/MovieUN-G/grounding/val_role.json',
        # 'test_data': '/data5/yzh/MovieUN_v2/MovieUN-G/grounding/test_movie.json',
        'model_saved_path': 'results/model_%s',
        'model_load_path': '/data5/yzh/MovieUN_v2/IANET/code/output/grounding/model-14',
        'max_num_words': 80,
        'max_num_nodes': 80,
        'max_num_frames': 200,
        'd_model': 512,
        'num_heads': 4,
        'batch_size': 64,
        'dropout': 0.2,
        'word_dim': 768,
        'frame_dim': 1024,
        'num_gcn_layers': 2,
        'num_attn_layers': 2,
        'display_n_batches': 50,
        'max_num_epochs': 20,
        'weight_decay': 1e-7,
        'lr_scheduler': 'inverse_sqrt',
        'lr': 8e-4
    }
    import argparse
    def dict_to_namespace(dictionary):
        namespace = argparse.Namespace()
        for key, value in dictionary.items():
            setattr(namespace, key, value)
        return namespace
    model_args = dict_to_namespace(model_args)

    model = Model(model_args)
    model.eval()
    model = model.to(torch.device('cuda:0'))
    model = torch.nn.DataParallel(model, device_ids=[0])
    model.load_state_dict(torch.load(model_args.model_load_path))

    predict_boxes, _, predict_scores, _, _ = model(**model_input)

    # select top 1 proposal
    predict_boxes = predict_boxes[0].cpu().detach().numpy()
    predict_scores = predict_scores[0].cpu().detach().numpy()

    max_idx = np.argmax(predict_scores)
    relative_start, relative_end = predict_boxes[max_idx]

    final_start = round(anchor_start + relative_start, 2)
    final_end = round(anchor_start + relative_end, 2)

    return final_start, final_end


def grounding_infer(movie_id, query):
    anchor_start, anchor_end = retrieve(movie_id, query)
    final_start, final_end = grounding(movie_id, anchor_start, anchor_end, query)

    return final_start, final_end

# if name == "__main__":
#     movie_id = '6965768652251628068'
#     query = '夏洛正在弹吉他'
#     anchor_start, anchor_end = retrieve(movie_id, query)
#     final_start, final_end = grounding(movie_id, anchor_start, anchor_end, query)

#     print(f'ANCHOR SPAN: [{anchor_start}, {anchor_end}]')
#     print(f'FINAL SPAN: [{final_start}, {final_end}]')