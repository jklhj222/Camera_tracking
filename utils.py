#!/usr/bin/env python3

import os
import json
import numpy as np
import base64
import cv2
import configparser

import DarknetFunc as DFUNC
import YoloObj

config = configparser.RawConfigParser()
config.read('config.txt')

# library of darknet
darknet_lib = config['DARKNET']['SHARED_LIB']

# configure for tracking system
darknet_model_dir = config['DARKNET_TRACK']['MODEL_DIR']
darknet_track_cfg = os.path.join(darknet_model_dir, 'test.cfg')
darknet_track_weights = os.path.join(darknet_model_dir, 'train_best.weights')
darknet_track_data = os.path.join(darknet_model_dir, 'task.data')
temp_img_file = os.path.join(darknet_model_dir, 'template.jpg')
temp_objs_coord_file = os.path.join(darknet_model_dir, 'template.txt')
temp_real_size = eval(config['DARKNET_TRACK']['TEMP_REAL_SIZE'])

cam_fov = eval(config['DARKNET_TRACK']['CAM_FOV'])
track_gpu = int(config['DARKNET_TRACK']['CUDA_IDX'])

# configure for detecting system
detect_model_dir = config['DARKNET_DETECT']['MODEL_DIR']
darknet_detect_cfg = os.path.join(detect_model_dir, 'test.cfg')
darknet_detect_weights = os.path.join(detect_model_dir, 'train_best.weights')
darknet_detect_data = os.path.join(detect_model_dir, 'task.data')

#darknet_detect_cfg = config['DARKNET_DETECT']['CFG']
#darknet_detect_weights = config['DARKNET_DETECT']['WEIGHTS']
#darknet_detect_data = config['DARKNET_DETECT']['DATA_FILE']
detect_gpu = int(config['DARKNET_DETECT']['CUDA_IDX'])

def ParseRecvJsonMsg(json_str):
#    print('json_str: ', json_str)
    json_obj = json.loads(json_str)

#    recv_id = json_obj['SendMsg']['id']
#    recv_img = json_obj['SendMsg']['img']
#    recv_check = json_obj['SendMsg']['checkid']
    recv_id = json_obj['id']
    recv_img = json_obj['img']
    recv_check = json_obj['checkid']
#    recv_timestamp = json_obj['timestamp']

#    return recv_id, recv_img, recv_check, recv_timestamp
    return recv_id, recv_img, recv_check
    

#def ParseSendJsonMsg(ID, timestamp, position_real=None, yaw=None, detect_objs=None):
def ParseSendJsonMsg(ID, plugin, position_real=None, yaw=None, detect_objs=None, defects=None):
    # string1 = '{"ReceiveMsg": [{"id": "2", "m_CamPos": {"x": "", "y": "", "z": "", "yaw": ""},'
    # string2 = '"m_AIResult": [{"Lx": "", "Ly": "", "Rx": "", "Ry": "", "confidence": "", "name": ""},'
    # string3 = '{"Lx": "", "Ly": "", "Rx": "", "Ry": "", "confidence": "", "name": ""}]}]}'

    output_dict = {}
    output_dict['ReceiveMsg'] = []
    output_dict['ReceiveMsg'].append({})

    ID = str(ID)
    output_dict['ReceiveMsg'][0]['id'] = ID
#    output_dict['id'] = ID

    output_dict['ReceiveMsg'][0]['plugin'] = plugin

#    output_dict['ReceiveMsg'][0]['timestamp'] = timestamp

    # for tracking 
    output_dict['ReceiveMsg'][0]['m_CamPos'] = {}
#    output_dict['m_CamPos'] = {}
    if position_real is not None:
        cam_x = str(int(position_real[0]))
        cam_y = str(int(position_real[1]))
        cam_z = str(int(position_real[2]))

        if yaw is not None:
            cam_yaw = '{:.2f}'.format(yaw)        
    
        else:
            cam_yaw = ''        
 
    else:
        cam_x = ''
        cam_y = ''
        cam_z = ''
        cam_yaw = ''        

    output_dict['ReceiveMsg'][0]['m_CamPos']['x'] = cam_x
    output_dict['ReceiveMsg'][0]['m_CamPos']['y'] = cam_y
    output_dict['ReceiveMsg'][0]['m_CamPos']['z'] = cam_z
    output_dict['ReceiveMsg'][0]['m_CamPos']['yaw'] = cam_yaw
#    output_dict['m_CamPos']['x'] = cam_x
#    output_dict['m_CamPos']['y'] = cam_y
#    output_dict['m_CamPos']['z'] = cam_z
#    output_dict['m_CamPos']['yaw'] = cam_yaw

    # for detecting
    output_dict['ReceiveMsg'][0]['m_AIResult'] = []
#    output_dict['m_AIResult'] = []
#    print('detect_objs: ', detect_objs)
    if detect_objs is not None: 
       if len(detect_objs) > 0:
            for obj in detect_objs:
                Lx = str(obj.l)
                Ly = str(obj.t)
                Rx = str(obj.r)
                Ry = str(obj.b)
                conf = str(obj.conf)
                name = obj.name 

                obj_dict = {'Lx': Lx, 'Ly': Ly, 'Rx': Rx, 'Ry': Ry, 
                            'confidence': conf, 'name': name}

                output_dict['ReceiveMsg'][0]['m_AIResult'].append(obj_dict)
#                output_dict['m_AIResult'].append(obj_dict)

    else:
        obj_dict = {'Lx': '', 
                    'Ly': '', 
                    'Rx': '', 
                    'Ry': '', 
                    'confidence': '',
                    'name': ''}

        output_dict['ReceiveMsg'][0]['m_AIResult'].append(obj_dict)
#        output_dict['m_AIResult'].append(obj_dict)

    output_dict['ReceiveMsg'][0]['detectpoints'] = []

    if defects is not None:
        for defect in defects:
            output_dict['ReceiveMsg'][0]['detectpoints'].append(defect) 

    output_json = json.dumps(output_dict)
        
    return output_json


def ParseSend3rdPartyJsonMsg(img_base64, padip, plugin, 
                             defects=None, position_real=None, yaw=None, report_doc_imgs=None):
    # img_base64: jpg image in base64 format
    # defects: defects dicts in a list, [{'id': 'id1', 'status': 'status1'}, 
    #                                    {'id': 'id2', 'status': 'status2'},
    #                                    ...]

    # exclude duplicated items
    if defects is not None:
        ids = []
        reduce_defects = []
        for defect in defects:
            if defect['id'] not in ids:
                reduce_defects.append(defect)
                ids.append(defect['id'])

    else:
        reduce_defects = None

    output_dict = {}
    output_dict['TAMsg'] = []
    output_dict['TAMsg'].append({})

#    output_dict['TAMsg'][0]['img'] = img_base64.decode('utf-8')
    output_dict['TAMsg'][0]['img'] = img_base64

    output_dict['TAMsg'][0]['padip'] = padip

    output_dict['TAMsg'][0]['plugin'] = plugin

    output_dict['TAMsg'][0]['detectpoints'] = []

    if reduce_defects is not None:
        for defect in reduce_defects:
            output_dict['TAMsg'][0]['detectpoints'].append(defect) 

    output_dict['TAMsg'][0]['m_CamPos'] = {}

    if position_real is not None:
        cam_x = str(int(position_real[0]))
        cam_y = str(int(position_real[1]))
        cam_z = str(int(position_real[2]))

        if yaw is not None:
            cam_yaw = '{:.2f}'.format(yaw)        
    
        else:
            cam_yaw = ''        
 
    else:
        cam_x = ''
        cam_y = ''
        cam_z = ''
        cam_yaw = ''        

    output_dict['TAMsg'][0]['m_CamPos']['x'] = cam_x
    output_dict['TAMsg'][0]['m_CamPos']['y'] = cam_y
    output_dict['TAMsg'][0]['m_CamPos']['z'] = cam_z
    output_dict['TAMsg'][0]['m_CamPos']['yaw'] = cam_yaw

    if report_doc_imgs is not None:
        output_dict['TAMsg'][0]['reportimgs'] = report_doc_imgs

    else:
        output_dict['TAMsg'][0]['reportimgs'] = []


    output_json = json.dumps(output_dict)

    return output_json


def ImgDetect(net, meta, img_np, thresh=0.25):
    # net: DFUNC.load_net(bytes(darknet_cfg, 'utf-8'),
    #                     bytes(darknet_weights, 'utf-8'), 0)
    # meta: DFUNC.load_meta(bytes(darknet_data, 'utf-8'))
    # img_np: image in numpy array 

    results = DFUNC.detect(net, meta, img_np, thresh)

    objs = []
    for result in results:
        obj = YoloObj.DetectedObj(result)
        objs.append(obj)

    # sort the objects by confidence
    objs = sorted(objs, key=lambda x: x.conf, reverse=True)
    
    return objs


def YoloTrackDetect(img_base64,
                    temp_img, 
                    net_track, meta_track,
                    net_detect, meta_detect):


    # decode image data from base64 format
    img_decode = base64.b64decode(img_base64)
    nparr = np.fromstring(img_decode, np.uint8)
    img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    print('input size: ', img_np.shape)


    # for tracking system
    DFUNC.set_gpu(track_gpu)
    track_objs = ImgDetect(net_track, meta_track, img_np)

    if len(track_objs) > 0:
        # exclude the ugly objects
        track_objs = [obj for obj in track_objs if obj.name != '1']
 
        # exclude the edge objects
        central_ratio = 0.6
        img_central_area = ( (int(img_np.shape[1]*(1-central_ratio)/2),
                              int(img_np.shape[0]*(1-central_ratio)/2)),
                             (int(img_np.shape[1]*(1-(1-central_ratio)/2)),
                              int(img_np.shape[0]*(1-(1-central_ratio)/2))) )
         
        track_objs = [ obj for obj in track_objs 
                       if obj.cx >= img_central_area[0][0] 
                       and obj.cy >= img_central_area[0][1] ]
 
        track_objs = [ obj for obj in track_objs 
                       if obj.cx <= img_central_area[1][0] 
                       and obj.cy <= img_central_area[1][1] ]
 
    cam_orient = YoloObj.CamOrient(track_objs,
                                   img_np.shape,
                                   cam_fov,
                                   temp_real_size,
                                   temp_img.shape,
                                   meta_track,
                                   temp_objs_coord_file)
 
    print('cam orientation: ', cam_orient.position_real,
                               cam_orient.yaw_deg)


    # for detecting system
    DFUNC.set_gpu(detect_gpu)
    detect_objs = ImgDetect(net_detect, meta_detect, img_np)
    if len(detect_objs) == 0:
        detect_objs = None

#    img = YoloObj.DrawBBox(track_objs, img_np)     
#    YoloObj.ShowImg(img)

#    print('detect_objs, track_objs: ', detect_objs, track_objs)
    return cam_orient, track_objs, detect_objs, img_np

def StartAI():
    # load yolo tracking model and template image info.
    DFUNC.set_gpu(track_gpu)
    net_track = DFUNC.load_net(bytes(darknet_track_cfg, 'utf-8'),
                               bytes(darknet_track_weights, 'utf-8'), 0)
    meta_track = DFUNC.load_meta(bytes(darknet_track_data, 'utf-8'))


    # load yolo detecting model
    DFUNC.set_gpu(detect_gpu)
    net_detect = DFUNC.load_net(bytes(darknet_detect_cfg, 'utf-8'),
                                bytes(darknet_detect_weights, 'utf-8'), 0)
    meta_detect = DFUNC.load_meta(bytes(darknet_detect_data, 'utf-8'))

    return net_track, meta_track, net_detect, meta_detect


def SaveReport(time_stamp, checkpt_id, cam_orient,
               track_objs, detect_objs, img_np):

    w_ratio = cam_orient.temp_tgt_ratio[0]
    h_ratio = cam_orient.temp_tgt_ratio[1]

    tgt_cx = cam_orient.tgt_cx
    tgt_cy = cam_orient.tgt_cy

    tgt_cam_x = cam_orient.xy_position_pixel[0]
    tgt_cam_y = cam_orient.xy_position_pixel[1]

    tgt_cam_realx = cam_orient.position_real[0]
    tgt_cam_realy = cam_orient.position_real[1]

    img = img_np
    img = YoloObj.DrawBBox(track_objs, img, width=3) 
    img = YoloObj.DrawBBox(detect_objs, img, color=(0,0,255), width=3) 
    img = cv2.putText(img,
                      'real pos ({}, {})'.format(int(tgt_cam_realx), 
                                                 int(tgt_cam_realy)),
                      (tgt_cx, tgt_cy-30),
                      cv2.FONT_HERSHEY_TRIPLEX,
                      0.5,
                      (0,0,255),
                      1,
                      cv2.LINE_AA)

    objs_temp = [] 
    report_objs = []
    if detect_objs is None:
        detect_objs = []

    for obj in detect_objs:
        tgtc2obj_vec = (obj.cx-tgt_cx, obj.cy-tgt_cy)

        obj_temp_position = ( tgt_cam_x + tgtc2obj_vec[0]*w_ratio,
                              tgt_cam_y + tgtc2obj_vec[1]*h_ratio )

        obj_temp_shape = ( obj.w * w_ratio, obj.h * h_ratio)

        objs_temp.append({obj.name: obj_temp_position})

        report_obj = ReportObj(obj.name, obj.conf, 
                               obj_temp_position, obj_temp_shape)

        report_objs.append(report_obj)

    report_dir = os.path.join('reports', time_stamp + '_reports')

    if not os.path.isdir(report_dir):
        os.makedirs(report_dir)

    report_file = os.path.join(report_dir, 
                               'check_point_{:03d}.txt'.format(int(checkpt_id)))

    report_img = os.path.join(report_dir, 
                               'check_point_{:03d}.jpg'.format(int(checkpt_id)))

    with open(report_file, 'w+') as f:
#        f.write('temp_cxy: ' + str(temp_cx) + '   ' + str(temp_cy), '\n')
        f.write('track_obj: ' + str(track_objs[0].name) + ' ' + str(track_objs[0].conf) + ' ' + str((track_objs[0].cx, track_objs[0].cy)) + '\n')
        f.write('cam_realpoition: ' +  str(cam_orient.position_real) + str(cam_orient.yaw_deg) + '\n')
        f.write('tgtc2obj_vec: ' + str(tgtc2obj_vec) + '\n')
        f.write('ratio: ' + str(w_ratio) + '   ' + str(h_ratio) + '\n')
        f.write(str(objs_temp) + '\n')

        YoloObj.SaveImg(img, save_path=report_img)

#    input('report saving test')   # for_test

    return report_objs

def ReduceFinalReport2(final_objs, overlap=0.75):
    reduce_objs = final_objs.copy()

    idx2 = 0
    for idx1, obj1 in enumerate(final_objs[:-1]):
        idx2 += 1
        for obj2 in final_objs[idx2:]:
            if obj2.a >= obj1.a:
                innerObj = obj1
                outerObj = obj2

            else:
                innerObj = obj2
                outerObj = obj1
  
            POI = YoloObj.ObjsPOI(innerObj, outerObj)
            print('idx2: ', idx1, idx2, (obj2.cx, obj2.cy))
            print('POI: ', POI)
            print()
            if POI >= overlap and obj2 in reduce_objs:
                reduce_objs.remove(obj2)

    return reduce_objs


def ReduceFinalReport(final_objs):
    reduce_objs = final_objs.copy()

    idx2 = 0
    for idx1, obj1 in enumerate(final_objs[:-1]):
        idx2 += 1
        for obj2 in reduce_objs[idx2:]:
            if obj2.r > obj1.cx > obj2.l and obj2.b > obj1.cy > obj2.t:
                reduce_objs.remove(obj2)

    return reduce_objs


class ReportObj():
    def __init__(self, obj_name, obj_conf, obj_temp_position, obj_temp_shape):
        self.name = obj_name
        self.conf = obj_conf
        self.cx = int(obj_temp_position[0])
        self.cy = int(obj_temp_position[1])
        self.w = obj_temp_shape[0]
        self.h = obj_temp_shape[1]
        self.l = int(obj_temp_position[0] - obj_temp_shape[0]/2)
        self.r = int(obj_temp_position[0] + obj_temp_shape[0]/2) 
        self.t = int(obj_temp_position[1] - obj_temp_shape[1]/2)
        self.b = int(obj_temp_position[1] + obj_temp_shape[1]/2)
        self.a = self.w * self.h 


#def ReadDefectLoc2(config, defect_loc_file):
#    # defect IDs on each check point
#    # CKPT0 = ((157, 85), (0, 1))
#    ckpts = [eval(config['CHECK_LIST'][ckpt]) for ckpt in config['CHECK_LIST']]
#
#    # read defect locations
#    defect_locs = {}
#    with open(defect_loc_file, 'r') as f:
#        for line in f.readlines():
#            defect_id = line.split(' ', 1)[0]
#            defect_loc = line.split(' ', 1)[1]
#
#            defect_locs[defect_id] = eval(defect_loc)
#
#    ckptc_defects_rel = {}
#    for ckpt_idx, ckpt in enumerate(ckpts):
#        idx_str = 'ckpt' + str(ckpt_idx)
#
#        defects = {}
#        for defect in ckpt[1]:
#            defects[str(defect)] = ((defect_locs[str(defect)][0]-ckpt[0][0], 
#                                     defect_locs[str(defect)][1]-ckpt[0][1]))
#
#        ckptc_defects_rel[idx_str] = defects
#
#
#    return ckpts, defect_locs, ckptc_defects_rel 


def ReadDefectLoc(config, defect_loc_file):
    # defect IDs on each check point
    # ex. CKPT0 = (0, 1)
    ckpt_defects = [ eval(config['CHECK_LIST'][ckpt]) 
                       for ckpt in config['CHECK_LIST'] ] 

    defect_real_locs = {}
    defect_type = {}
    with open(defect_loc_file, 'r') as f:
        for line in f.readlines():
            defect_id = line.split(' ', 1)[0]
            defect_loc = line.split(' ', 1)[1].rsplit(' ', 1)[0]
            defect_define_type = line.rsplit(' ', 1)[1]            

            defect_real_locs[defect_id] = eval(defect_loc)
            defect_type[defect_id] = defect_define_type.strip('\n')

    print('ckpt_defects: ', ckpt_defects)   # for_test
    print('defect_type: ', defect_type)   # for_test

    return ckpt_defects, defect_real_locs, defect_type



def Gen3rdpCkptDictStr(config, throu_ckpt, ckpt_id, cam_orient, detect_objs):
    # read info. about ckeck points and defects
#    ckpts, defect_locs, ckptc_defects_rel = \
#            ReadDefectLoc(config, 'defects_location.txt')

    ckpt_defects, defect_real_locs, defect_type = \
            ReadDefectLoc(config, 'defects_location.txt')

#    ckpt_str = 'ckpt' + str(throu_ckpt)
    ckpt_str = 'ckpt' + ckpt_id
    cam_real_pos = cam_orient.position_real
    mm2pixel = cam_orient.mm2pixel

    if mm2pixel is not None:
#    defect_locs = []
#    for defect_id in ckpts[throu_ckpt][1]:
#        defect_loc = ( (int(cam_orient.tgt_cx + 
#                            ckptc_defects_rel[ckpt_str][str(defect_id)][0] * 
#                            cam_orient.mm2pixel[0]),
#                        int(cam_orient.tgt_cy + 
#                            ckptc_defects_rel[ckpt_str][str(defect_id)][1] *
#                            cam_orient.mm2pixel[1])) )
#
#        defect_locs.append(defect_loc)
        defect_locs = []
        defect_sizes = []
        for defect_id in ckpt_defects[int(ckpt_id)]:
            defect_loc = ( int(cam_orient.tgt_cx + 
                               (defect_real_locs[str(defect_id)][0] -
                                cam_real_pos[0]) * mm2pixel[0]),
                           int(cam_orient.tgt_cy + 
                               (defect_real_locs[str(defect_id)][1] - 
                                cam_real_pos[1]) * mm2pixel[1]) )

            defect_size = ( int(defect_real_locs[str(defect_id)][2] * mm2pixel[0]),
                            int(defect_real_locs[str(defect_id)][3] * mm2pixel[1]) )

            defect_locs.append(defect_loc)
            defect_sizes.append(defect_size)

        out_dicts = []
        for defect_idx, defect_loc in zip(ckpt_defects[int(ckpt_id)], defect_locs):        

            count = 0
            for obj in detect_objs: 
#                if obj.l-50 <= defect_loc[0] <= obj.r+50 and obj.t-50 <= defect_loc[1] <= obj.b+50:
                if obj.l <= defect_loc[0] <= obj.r and obj.t <= defect_loc[1] <= obj.b:
#                    dict_str.append({'id': str(defect_idx), 'status': 'NOK'})
                    out_dicts.append({'id': str(defect_idx), 
                                      'status': defect_type[str(defect_idx)]})
                    break

                else:
                    count += 1

                if count == len(detect_objs):
                    out_dicts.append({'id': str(defect_idx), 'status': 'OK'})


#        return str(dict_str), ckpts[throu_ckpt][1], defect_locs
        return out_dicts, ckpt_defects, defect_locs, defect_sizes

    else:
        return []


def Gen3rdpReportJson(cam_orient, report_dir, ckpt_imgs, ckpt_defects, allOut_dicts, 
                      all_defect_locs, all_defect_sizes, padip, plugin):
    # ckpt_imgs: a list which stores images of each check point
    # ckpt_defects: [(0, 1, 2), (3, 4), (5, 6, 7), ...]; from config.txt
    # allOut_dicts: [ {'id': '0', 'status': 'OK'}, 
    #                 {'id': '1', 'status': 'miss_screw'}, ... ]; state of every defect
    # all_defect_locs: [(200, 500), (350, 180), ...]; position (pixel) of defects in each check point
    # all_defect_sizes: [(50, 30), (70, 30), ...]; 
    #                   defect bbox size (width, height in pixel) of defects in each check point

    import copy

    report_doc_imgs = []

    i = 0   # for_test
    for img_idx, img in enumerate(ckpt_imgs):
        for defect in ckpt_defects[img_idx]:
            img_tmp = copy.deepcopy(img)

            defect_status = allOut_dicts[defect]['status']

            defect_l = int( all_defect_locs[defect][0] - 
                            all_defect_sizes[defect][0]/2 )
            defect_r = int( all_defect_locs[defect][0] + 
                            all_defect_sizes[defect][0]/2 )
            defect_t = int( all_defect_locs[defect][1] - 
                            all_defect_sizes[defect][1]/2 )
            defect_b = int( all_defect_locs[defect][1] + 
                            all_defect_sizes[defect][1]/2 )

            if defect_status == 'OK':
                cv2.rectangle( img_tmp,
                               (defect_l, defect_t),
                               (defect_r, defect_b),
                               (0, 255, 0), 5 ) 

            else:
                cv2.rectangle( img_tmp,
                               (defect_l, defect_t),
                               (defect_r, defect_b),
                               (0, 0, 255), 5 )

            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 20]
            img_tmp_enc = cv2.imencode('.jpg',
                                       img_tmp,
                                       encode_param)[1].tostring()

            img_tmp_b64 = base64.b64encode(img_tmp_enc).decode('utf-8')
    
            report_doc_imgs.append({'id': str(defect), 'img': img_tmp_b64})

            print('report_dir: ' + '{}/defect_img{:03d}.jpg'.format(report_dir, i))   # for_test
            cv2.imwrite('{}/defect_img{:03d}.jpg'.format(report_dir, i), img_tmp)   # for_test
            i += 1   # for_test

             
    report_3rdp_json = ParseSend3rdPartyJsonMsg('',
                                                padip,
                                                plugin,
                                                allOut_dicts,
                                                cam_orient.position_real,
                                                cam_orient.yaw_deg,
                                                report_doc_imgs)

    return report_3rdp_json


def FixTaskdataLabelNames(file_path, label_names_file=None):
    if label_names_file is None:
        label_names_file = file_path.replace(os.path.basename(file_path), 'label.names')

    new_lines = []

    with open(file_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        if line.strip() != '' and line.strip().startswith('names'):
            key, value = line.split('=')

            new_lines.append(key.strip() + '=' + label_names_file + '\n')

        else:
            new_lines.append(line)

            with open(file_path, 'w') as f:
                f.writelines(new_lines)


