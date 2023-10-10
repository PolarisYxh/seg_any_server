# -- coding: utf-8 --**
import json
import base64
import subprocess
import os
import traceback
import logging
from flask import send_from_directory
import cv2

def readjson(file):
    with open(file, 'r', encoding="utf-8") as load_f:
        load_dict = json.load(load_f)
    return load_dict
def writejson(file, write_dict):
    with open(file, "w", encoding="utf-8") as dump_f:
        json.dump(write_dict, dump_f, ensure_ascii=False)

import numpy as np


def cvmat2base64(img_np, houzhui='.jpg'):
    #opencv的Mat格式转为base64
    image = cv2.imencode(houzhui, img_np)[1]
    base64_data = str(base64.b64encode(image))
    return base64_data[2:-1]


def base642cvmat(base64_data):
    #base64转为opencv的Mat格式
    imgData = base64.b64decode(base64_data)
    nparr = np.frombuffer(imgData, np.uint8)
    img_np = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
    return img_np
class Handler1(object):
    def __init__(self,rFolder,bDebug,log):
        self.log = log
        self.log.logger.info('#########################segmentall Handler init##########################')
        ## TODO: put own function module
        from pytorch_hair_segmentation.interface import HairSegment
        ## model init
        load_dict = readjson(os.path.dirname(__file__) + "/config.json")
        device=load_dict["device"]
        if device=="cuda":
            os.environ['CUDA_VISIBLE_DEVICES'] =load_dict['gpus']
        self.app = HairSegment(device)
        self.cache_path = os.path.join(rFolder,'cache')
        logging.info('segmentall handler init done.')
        self.log.logger.info('#########################segmentall Handler init done#######################')
    def queue_callback(self, mode, json_data):
        reqCode = json_data['reqCode']
        self.log.logger.info(f'{reqCode}:#################Handler start#################.')
        self.log.logger.info(f'{reqCode}: mode={mode}')
        response_data = {'reqCode': reqCode, 'error': 0, 'errorInfo': ''}
        try:
            ## TODO:
            rst = self.process(json_data)
            # rst=json.dumps(rst,cls=MyEncoder)
            response_data = {'reqCode': reqCode, 'masks': rst, 'error':0, 'errorInfo': ''}
            logging.info(f'Handler successfully, reqCode: {reqCode}.')
            os.system(f'echo \"Handler successfully\" >> cache/{reqCode}_service_log.log')

        except Exception as ex:
            response_data = {'reqCode': reqCode, 'error':-1, 'errorInfo': str(ex)}
            logging.error(f'Handler fail ... reqCode: {reqCode} , {traceback.format_exc()}')
            os.system(f'echo \"Handler fail ... reqCode: {reqCode} , {traceback.format_exc()}\" >> cache/{reqCode}_service_log.log')

        finally:
            logging.info(f'Handler end, reqCode: {reqCode}.')
            return response_data

    def process(self, json_data):
        reqCode = json_data['reqCode']
        img = base642cvmat(json_data['imgFile'])
        # cv2.imshow("1",img)
        # cv2.waitKey()
        masks = self.app.get_segment(img,reqCode)
        masks[masks>0]=1
        masks  = masks.astype('bool')
        return masks.tolist()
    
class Handler(object):
    def __init__(self,rFolder,bDebug,log):
        self.log = log
        self.log.logger.info('#########################segmentall Handler init##########################')
        ## TODO: put own function module
        from get_segment import SAM
        ## model init
        load_dict = readjson(os.path.dirname(__file__) + "/config.json")
        model_type = load_dict["model_type"]
        device=load_dict["device"]
        if device=="cuda":
            os.environ['CUDA_VISIBLE_DEVICES'] =load_dict['gpus']
        self.app = SAM(model_type,device)
        self.cache_path = os.path.join(rFolder,'cache')
        logging.info('segmentall handler init done.')
        self.log.logger.info('#########################segmentall Handler init done#######################')
    def queue_callback(self, mode, json_data):
        reqCode = json_data['reqCode']
        self.log.logger.info(f'{reqCode}:#################Handler start#################.')
        self.log.logger.info(f'{reqCode}: mode={mode}')
        response_data = {'reqCode': reqCode, 'error': 0, 'errorInfo': ''}
        try:
            if mode=="img":
                ## TODO:
                rst = self.process(json_data)
                # rst=json.dumps(rst,cls=MyEncoder)
                response_data = {'reqCode': reqCode, 'masks': rst, 'error':0, 'errorInfo': ''}
                logging.info(f'Handler successfully, reqCode: {reqCode}.')
                os.system(f'echo \"Handler successfully\" >> cache/{reqCode}_service_log.log')

        except Exception as ex:
            response_data = {'reqCode': reqCode, 'error':-1, 'errorInfo': str(ex)}
            logging.error(f'Handler fail ... reqCode: {reqCode} , {traceback.format_exc()}')
            os.system(f'echo \"Handler fail ... reqCode: {reqCode} , {traceback.format_exc()}\" >> cache/{reqCode}_service_log.log')

        finally:
            logging.info(f'Handler end, reqCode: {reqCode}.')
            return response_data

    def process(self, json_data):
        reqCode = json_data['reqCode']
        img = base642cvmat(json_data['imgFile'])
        # cv2.imshow("1",img)
        # cv2.waitKey()
        masks = self.app.get_mask(img,np.array(json_data['point_coords']),np.array(json_data['point_labels']))

        
        # for i,face in enumerate(masks):
        #     face_dict = {}
        #     for k in face:
        #         if isinstance(face[k], np.integer):
        #             face_dict[k]=int(face[k])
        #         elif isinstance(face[k], np.floating):
        #             face_dict[k]=float(face[k])
        #         elif isinstance(face[k], np.ndarray):
        #             face_dict[k]=face[k].tolist()
        #         else:
        #             face_dict[k]=face[k]
        #     # face_dict['normed_embedding'] = face.normed_embedding.tolist()
        #     result["face"+f"_{i}"]=face_dict
        return masks.tolist()


def delete_tmp_file(reqCode, add_files_list=[]):
    cache_file_list = [f"cache/{reqCode}_service_log.log"]
    cache_file_list += add_files_list
    for fn in cache_file_list:
        subprocess.call(['rm', fn])

def file_b64encode(file_path):
    with open(file_path, 'rb') as f:
        file_string = f.read()
    file_b64encode = str(base64.b64encode(file_string), encoding='utf-8')
    return file_b64encode

def file_b64decode(file_Base64, file_path):
    missing_padding = len(file_Base64)%4
    if missing_padding!=0:
        file_Base64 += ('='*(4-missing_padding))
    file_encode = base64.b64decode(file_Base64)
    with open(file_path, 'wb') as f:
        f.write(file_encode)
    return

def download_reqcode_log(reqCode):
    log_path = './cache'
    filename = f'{reqCode}_service_log.log'
    file_path = os.path.join(log_path, filename)
    if os.path.exists(file_path):
        return send_from_directory(log_path, filename=filename, as_attachment=True)
    else:
        return f'{file_path} does not exist. Service requests sucessfully, or {reqCode} service does not request.'
