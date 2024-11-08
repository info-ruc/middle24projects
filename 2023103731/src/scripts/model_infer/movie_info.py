import os
import json
import h5py
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# TODO provide more information
def actor_info(movie_id, begin, end, user_input):
    metadata = os.path.join('/data4/zq/Movies_dataset/code/gen_anno/video_shots/metadata/xigua_intro', movie_id+'.json')
    face_fea_file = '/data4/myt/MovieChat/features/face_profile_512dim.hdf5'
    movie_face_fea_file = os.path.join('/data5/yzh/DATASETS/Movie101/feature/frame_face', movie_id+'.mp4.npy.npz')
    
    
    movie_data = np.load(movie_face_fea_file)
    face_feature = movie_data['features'][begin].reshape(3, 512)
    
    face_fea = h5py.File(face_fea_file, 'r')
    
    result_ids = []
    for face in face_feature:
        if np.sum(face) != 0: # identify a face
            mx_sim, result_id = 0, ''
            for role_face_id in face_fea.keys():
                comp_fea = face_fea[role_face_id]['features'][:]
                sim = cosine_similarity(comp_fea.reshape(1, -1), face.reshape(1, -1))[0][0]
                if sim > mx_sim:
                    mx_sim, result_id = sim, role_face_id
            result_ids.append(result_id)
            
    with open(metadata, 'r') as f:
        meta_info = json.load(f)
    
    result = []
    for cel_id in result_ids:
        for actor in meta_info["actorList"]:
            if actor["celebrityId"] == cel_id:
                result.append({"actor": actor["name"], "role": actor["roleName"]})
    
    return result

def movie_intro(movie_id):
    metadata = os.path.join('/data4/zq/Movies_dataset/code/gen_anno/video_shots/metadata/xigua_intro', movie_id+'.json')
    with open(metadata, 'r') as f:
        meta_info = json.load(f)
    return {"名称": meta_info["title"], "上映时间":meta_info["year"], "介绍":meta_info["intro"], "标签":meta_info["tagList"], "地区": meta_info["areaList"]}
