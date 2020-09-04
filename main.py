#!/usr/bin/env python3

import socket
import SocketSanta 
import DarknetFunc as DFUNC
import YoloObj
import utils 

import os
import time
import multiprocessing as mp 
from multiprocessing.managers import BaseManager
import configparser
import numpy as np
import cv2
import base64
import json
import copy
import psutil

config = configparser.RawConfigParser()
config.read('config.txt')

# library of darknet
darknet_lib = config['DARKNET']['SHARED_LIB']

# configure for tracking system
track_model_dir = config['DARKNET_TRACK']['MODEL_DIR']
darknet_track_cfg = os.path.join(track_model_dir, 'test.cfg')
darknet_track_weights = os.path.join(track_model_dir, 'train_best.weights')
darknet_track_data = os.path.join(track_model_dir, 'task.data')
temp_img_file = os.path.join(track_model_dir, 'template.jpg')
temp_objs_coord_file = os.path.join(track_model_dir, 'template.txt')
temp_real_size = eval(config['DARKNET_TRACK']['TEMP_REAL_SIZE'])
cam_fov = eval(config['DARKNET_TRACK']['CAM_FOV'])
track_gpu = int(config['DARKNET_TRACK']['CUDA_IDX'])

utils.FixTaskdataLabelNames(darknet_track_data)

# configure for detecting system
detect_model_dir = config['DARKNET_DETECT']['MODEL_DIR']
darknet_detect_cfg = os.path.join(detect_model_dir, 'test.cfg')
darknet_detect_weights = os.path.join(detect_model_dir, 'train_best.weights')
darknet_detect_data = os.path.join(detect_model_dir, 'task.data')
detect_gpu = int(config['DARKNET_DETECT']['CUDA_IDX'])

utils.FixTaskdataLabelNames(darknet_detect_data)

# configure for socket connect
host = config['SOCKET_CONNTECT']['HOST']
track_ports = eval(config['SOCKET_CONNTECT']['TRACK_PORTS'])
thirdp_port = eval(config['SOCKET_CONNTECT']['THIRDP_PORT'])
max_receive = int(config['SOCKET_CONNTECT']['MAX_RECEIVE'])

# read template image file
temp_img = cv2.imread(temp_img_file)

net_track,meta_track,net_detect,meta_detect = utils.StartAI()
print('AI ready')

#########################
#     Restart point     #
#########################

def run():
    global conn_3rdp

    # connect to pad with socket
    recv_conn = send_conn = None
    while recv_conn is None and send_conn is None:
        recv_conn, recv_addr, send_conn, send_addr = \
          SS.accept(s_recv, s_send)
 
    padip = recv_addr[0]
  
    print('pad accept done: ', recv_addr, send_addr)   # for_test

    # create report directory
    time_stamp = time.strftime("%Y-%m-%d_%H%M%S", time.localtime())
    report_dir = os.path.join('reports', time_stamp + '_reports')
    os.makedirs(report_dir)
    result_pic_dir = os.path.join(report_dir, 'results')
    os.makedirs(result_pic_dir)
    final_report = []
    total_time = 0
    ii = 0
    throu_ckpt = -1
    startup = False
    allOut_dicts = []
    all_defect_locs = []
    all_defect_sizes = []
    ckpt_imgs = []
    report_doc_imgs = []

    # check charging status of laptop
    battery = psutil.sensors_battery()
    charging = battery.power_plugged
    plugin = '0' if charging == False else '1'

    while True:
        print()

        ## start pad process ##
        # get the data length from pad
        process1 = mp.Process(name='get_data',
                              target=SS.get_data,
                              args=(recv_conn, 10, q))

        process1.daemon = True
        process1.start()

        head = q.get()
        data_len = int(SS.bytes2str(head).strip('\x00'))
        
        # for_test
        init_time = time.time()

        count = data_len
        data_str = ''

        while count > 0:
            print('count: ', count) # for_test

            # get data from pad
            process1 = mp.Process(target=SS.get_data,
                                  args=(recv_conn, count, q))

            process1.daemon = True
            process1.start()

            data = q.get() 
            data_decode = SS.bytes2str(data)

            data_str += data_decode

            count -= len(data)
            if count == 0:
                # get info. from pad
                recv_id, img_base64, ckpt_id = \
                  utils.ParseRecvJsonMsg(data_str)

                print('recv_id: ', recv_id)
                print('ii: ', ii)

                # start AI
                if recv_id == '1':
                    AI_ready_json = utils.ParseSendJsonMsg(recv_id, plugin)
                    print('AI_ready_json', AI_ready_json)

                    # send the length info. of data to 3rd-p program
                    # (10 bytes: ex. '123456\x00\x00\x00\x00')
                    output_len_str = str(len(AI_ready_json))
                    send_conn.send( (output_len_str +
                                    '\x00'*(10-len(output_len_str))).encode() )

                    send_conn.send( AI_ready_json.encode() )
                    startup = True

                # start scanning
                elif recv_id in ['2', '2_diag', '3']:
                    AI_init_time = time.time()   # for_test


                    cam_orient, track_objs, orig_detect_objs, img_np = \
                      utils.YoloTrackDetect(img_base64, temp_img,
                                            net_track, meta_track,
                                            net_detect, meta_detect)

                    # exclude self_diag_abnormal objects
                    if orig_detect_objs is not None:
                        detect_objs = [obj for obj in orig_detect_objs 
                                         if obj.name != 'self_diag_abnormal']

                    else:
                        detect_objs = None

                    print('padip: ', padip)   # for_test
                    print('detect objs: ', detect_objs)   # for_test
                    print('plugin: ', plugin)   # for_test
                    detect_json = utils.ParseSendJsonMsg(recv_id,
                                                         plugin,
                                                         cam_orient.position_real,
                                                         cam_orient.yaw_deg,
                                                         detect_objs,
                                                         allOut_dicts)
                    print('detect_json: ', detect_json)   # for_test
#                    # for_test
#                    if detect_objs is not None:
#                        for obj in detect_objs:
#                            print(obj.name, obj.conf)

                    # self diagnosis
                    if recv_id == '2_diag':
                        if cam_orient.mm2pixel is not None:
                            img_np_cx = int(img_np.shape[1]/2)
                            img_np_cy = int(img_np.shape[0]/2)
                    
                            if detect_objs is not None:
                                diag_objs = [ obj for obj in orig_detect_objs 
                                                if obj.name == 'self_diag_abnormal' ]
                    
                            else:
                                diag_objs = []
                    
                            diag_objs_pos = []
                            for obj in diag_objs:
                    
                                obj_rel_pos = ( (obj.cx - img_np_cx) / cam_orient.mm2pixel[0], 
                                                (obj.cy - img_np_cy) / cam_orient.mm2pixel[1] )
                    
                                obj_real_pos = ( cam_orient.position_real[0] + obj_rel_pos[0],
                                                 cam_orient.position_real[1] + obj_rel_pos[1] )
                    
                                diag_objs_pos.append(obj_real_pos)
                    
                            diag_objs_exist = [ obj for obj in diag_objs_pos 
                                                  if 0 <= obj[0] <= temp_real_size[1] 
                                                  and 0 <= obj[1] <= temp_real_size[0] ]
                    
                    
#                            diag_id == '2_diag' if len(diag_objs_exist) != 0 else '2_diag_none'
                            if len(diag_objs_exist) != 0 and cam_orient.position_real is not None:
                                diag_id = recv_id
                    
                            else:
                                diag_id = '2_diag_none'

                        else:
                            diag_id = recv_id
                            diag_objs_exist = []
                            diag_objs = []

                        print('diag test: ', diag_objs_exist)   # for_test
                        print('diag recv_id: ', diag_id)   # for_test

                        detect_json = utils.ParseSendJsonMsg(diag_id,
                                                             plugin,
                                                             cam_orient.position_real,
                                                             cam_orient.yaw_deg,
                                                             diag_objs,
                                                             allOut_dicts)


                    # through check point and save checkpoint
                    if recv_id == '3':
                        # save images of each check point
                        ckpt_imgs.append(img_np)

                        report_objs = utils.SaveReport(time_stamp, ckpt_id,
                                                       cam_orient, track_objs,
                                                       detect_objs, img_np)
                     
                        final_report.extend(report_objs)

                        throu_ckpt += 1

                        # not through any check point yet
                        if throu_ckpt < 0:
#                            dict_str = str([])
                            out_dicts = []
                
                        # through check point
                        else :
                            if detect_objs is None:
                                out_dicts, ckpt_defects, defect_locs, defect_sizes = \
                                  utils.Gen3rdpCkptDictStr(config,
                                                           throu_ckpt,
                                                           ckpt_id,
                                                           cam_orient,
                                                           [])
                
                            else:
                                out_dicts, ckpt_defects, defect_locs, defect_sizes = \
                                  utils.Gen3rdpCkptDictStr(config,
                                                           throu_ckpt,
                                                           ckpt_id,
                                                           cam_orient,
                                                           detect_objs)
                
                            allOut_dicts.extend(out_dicts)
                            all_defect_locs.extend(defect_locs)
                            all_defect_sizes.extend(defect_sizes)
                            
                        # for_test
                        with open('ckpt_report_' + str(throu_ckpt) + '.txt', 'w') as f:
                            f.write( str(out_dicts) + ' ' + str(len(out_dicts)) )
                 
#                        input('stop in recv_id = 3: ')   # for_test

                    print('throu_ckpt: ', throu_ckpt)   # for_test

                    # send the length info. of data to 3rd-p program
                    # (10 bytes: ex. '123456\x00\x00\x00\x00')
                    output_len_str = str(len(detect_json))
                    send_conn.send( (output_len_str +
                                    '\x00'*(10-len(output_len_str))).encode() )
            
                    send_conn.send( detect_json.encode() )
                    print('AI time: ', time.time() - AI_init_time, 's')   # for_test

                elif recv_id == '4':
                    allOut_dicts.sort(key=lambda x: int(x['id']))

                    # send final report images to 3rd party program
                    if conn_3rdp is not None:
                        report_3rdp_json = \
                          utils.Gen3rdpReportJson(cam_orient, report_dir,
                                                  ckpt_imgs, ckpt_defects,
                                                  allOut_dicts,
                                                  all_defect_locs, 
                                                  all_defect_sizes,
                                                  padip,
                                                  plugin)

                        # send the length info. of data to 3rd-p program
                        # (10 bytes: ex. '123456\x00\x00\x00\x00')
                        output_len_str = str(len(report_3rdp_json))
                        conn_3rdp.send( (output_len_str +
                                        '\x00'*(10-len(output_len_str))).encode() )
  
                        # send data to 3rd party program
                        conn_3rdp.send( report_3rdp_json.encode() )                            

                    closeAI_json = utils.ParseSendJsonMsg(recv_id, plugin)
                    send_conn.send( closeAI_json.encode() )

                elif recv_id == '5':

                    final_img = os.path.join(report_dir, 'final_report.jpg')

                    report_img = YoloObj.DrawBBox(final_report, temp_img)

                    YoloObj.SaveImg(report_img, save_path=final_img)


                    # send report to pad
                    report_json = utils.ParseSendJsonMsg(recv_id,
                                                         plugin,
                                                         detect_objs=final_report)

                    send_conn.send( report_json.encode() )

                    raise ValueError 

                interval = time.time() - init_time
                total_time += interval
                ii += 1

                average_time = total_time / ii
                print('perframe: ', interval, 's')   # for_test
                print('average_time: ', average_time, 's')   # for_test
#                f.write(str(ii) + '   ' + str(perframe) + '\n') # for_test

                ## end pad process ##

      
                print('startup : ', startup)   # for_test
                ## start 3rd party program process ##
                # get status of 3rd party connection
                print('3rd-p status: ', inst.get())
                if conn_3rdp is None:
                    print('not get 3rd status yet.')
                    conn_3rdp, addr_3rdp = inst.get()
             
                elif conn_3rdp is not None and startup == True:
                    init_3rdp_time = time.time()   
                    img_3rdp = YoloObj.DrawBBox(detect_objs, img_np, color=(0, 0, 255))
#                    img_3rdp = YoloObj.DrawBBox(track_objs, img_3rdp, color=(0, 255, 0))   # for_test
             
                    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 20]
                    img_3rdp_enc = cv2.imencode('.jpg', 
                                                img_3rdp, 
                                                encode_param)[1].tostring()
                    
                    img_3rdp_b64 = base64.b64encode(img_3rdp_enc).decode('utf-8')
             
             
             
                    print('allOut_dicts2: ', allOut_dicts)
                    out_3rdp_json = utils.ParseSend3rdPartyJsonMsg(img_3rdp_b64,
                                                                   padip,
                                                                   plugin,
#                                                                   eval(dict_str),
                                                                   allOut_dicts,
                                                                   cam_orient.position_real,
                                                                   cam_orient.yaw_deg)
             
             
#                    print('out_dicts: ', out_dicts)   # for_test
#                    print('allOut_dicts: ', allOut_dicts)   # for_test
             
                    try:
                        # send the length info. of data to 3rd-p program
                        # (10 bytes: ex. '123456\x00\x00\x00\x00')
                        print('test 3rdp')   # for_test
                        output_len_str = str(len(out_3rdp_json))
                        conn_3rdp.send( (output_len_str +
                                        '\x00'*(10-len(output_len_str))).encode() )
             
#                        print('test true out_3rdp_json: ', (out_3rdp_json) )  # for_test
             
                        # send data to 3rd party program
                        conn_3rdp.send( out_3rdp_json.encode() )                            
                        print('3rdp time: ', time.time() - init_3rdp_time, 's')   # for_test
             
#                    except socket.error, ConnectionResetError as sockerr:
                    except:
#                        print('sockerr: ', sockerr)   # for_test
#                        input('enter to continue')   # for_test
                        inst.set(None, None) 
                        conn_3rdp, addr_3rdp = inst.get()
             
                elif conn_3rdp is not None and startup == False:
                    init_3rdp_time = time.time()   
                    out_3rdp_json = utils.ParseSend3rdPartyJsonMsg('', padip, plugin)
             
                    try:
                        # send the length info. of data to 3rd-p program
                        # (10 bytes: ex. '123456\x00\x00\x00\x00')
                        output_len_str = str(len(out_3rdp_json))
                        conn_3rdp.send( (output_len_str +
                                        '\x00'*(10-len(output_len_str))).encode() )
             
                        # send data to 3rd party program
                        conn_3rdp.send( out_3rdp_json.encode() )                            
                        print('test false out_3rdp_json: ', (out_3rdp_json) )  # for_test
                        print('3rdp time: ', time.time() - init_3rdp_time, 's')   # for_test
             
#                    except socket.error, ConnectionResetError as sockerr:
                    except:
#                        print('sockerr: ', sockerr)   # for_test
#                        input('enter to continue')   # for_test
                        inst.set(None, None) 
                        conn_3rdp, addr_3rdp = inst.get()
             
                    ## end 3rd party program process ##


def socket_connect(host, track_ports, thridp_port):
    # create socket objects for pad and 3rd party program
    SS = SocketSanta.SocketSanta(host, track_ports)
    S3P = SocketSanta.Socket3rdParty(host, thirdp_port)
  
    s_recv, s_send = SS.connect(SS.host, SS.recv_port, SS.send_port)

    # set 3rd party program to sub-processing
    BaseManager.register('socket_status', SocketSanta.SocketStatus)
    manager = BaseManager()
    manager.start()
    inst = manager.socket_status()
    print('inst: ', inst, type(inst))   # for_test

    proc_3rdp = mp.Process(target=S3P.connect, args=(inst, ))
    proc_3rdp.daemon = True
    proc_3rdp.start()
    conn_3rdp, addr_3rdp = inst.get()
    print('conn_3rdp, addr_3rdp: ', conn_3rdp, addr_3rdp)   # for_test

    return SS, S3P, s_recv, s_send, conn_3rdp, inst    

if __name__ == '__main__':
    q = mp.Queue()

    print(host, track_ports, thirdp_port)

    SS, S3P, s_recv, s_send, conn_3rdp, inst = \
      socket_connect(host, track_ports, thirdp_port)

    while True:            
        try:
            run()

#        except (ValueError, OSError, ConnectionResetError):
        except:
            print('except in main program1')   # for_test

            # send message to 3rd program before disconnect
            if conn_3rdp is not None:
                discon_3rdp_json = utils.ParseSend3rdPartyJsonMsg('disconnect', '0.0.0.0', '1')
 
                try:
                    # send the length info. of data to 3rd-p program
                    # (10 bytes: ex. '123456\x00\x00\x00\x00')
                    output_len_str = str(len(discon_3rdp_json))
                    conn_3rdp.send( (output_len_str +
                                    '\x00'*(10-len(output_len_str))).encode() )
                    
                    # send data to 3rd party program
                    conn_3rdp.send( discon_3rdp_json.encode() )                            

                except:
                    pass

            SS.disconnect(s_recv, s_send)
            S3P.disconnect()

            SS, S3P, s_recv, s_send, conn_3rdp, inst = \
              socket_connect(host, track_ports, thirdp_port)

            print('waiting for next round detection.')
#            inst.set(None, None) 
#            conn_3rdp, addr_3rdp = inst.get()


#        try:
#            run()
#
#        except socket.error:
#            SS.disconnect(s_recv, s_send)
#            S3P.disconnect() 

