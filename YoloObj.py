#!/usr/bin/env python3
""" Created on Wed Jan 23 16:03:43 2019 @author: jklhj """
from math import ceil, sqrt, pi, sin, tan, acos, pow
import numpy as np
import sys 
import cv2 

class DetectedObj():
    def __init__(self, result):
        self.name = result[0].decode('utf-8')
        self.conf = int(round(result[1]*100))
        self.bbox_yolo = result[2]

        self.bbox = self.calc_bbox(result[2][0],
                                   result[2][1],
                                   result[2][2],
                                   result[2][3])

        self.cx = int(result[2][0])
        self.cy = int(result[2][1])
        self.l = self.bbox[0]
        self.r = self.bbox[1]
        self.t = self.bbox[2]
        self.b = self.bbox[3]
        self.w = self.bbox[4]
        self.h = self.bbox[5]
        self.a = self.bbox[6]

        self.obj_string = '{"' + self.name + \
                          '":' + str(self.conf) + \
                          ',"left":'  + str(self.l) + \
                          ',"right":' + str(self.r) + \
                          ',"top":'   + str(self.t) + \
                          ',"bot":'   + str(self.b) + '}'
   
 
    def calc_bbox(self, x, y, w, h):
        left   = int(x - w/2)
        right  = int(x + w/2)
        top    = int(y - h/2)
        bottom = int(y + h/2)
        width  = right - left
        height = bottom - top
        area   = width * height
        
        return (left, right, top, bottom, width, height, area)

    def CalcInOutObj(inout_pairs, poi_thresh, objs):
        all_POIs = []
        all_innerObjs = []
#        out = []
        OutObjs = []
        for innerName, outerName in inout_pairs:
            if innerName == outerName:
                print('\nThe inner object and outer object are the same, '
                      'are you serious? '
                      'Check the config.txt again, OK?')
             
                sys.exit()

            innerObjs = [obj for obj in objs if obj.name == innerName]
            outerObjs = [obj for obj in objs if obj.name == outerName]
            print('innerObjs: ', innerObjs, len(innerObjs))
            print('outerObjs: ', outerObjs, len(outerObjs))

            num_outer = len(outerObjs)
    
            POIs = [ ObjsPOI(inner, outer) 
                       for inner in innerObjs
                       for outer in outerObjs ]
    
            all_POIs.append( (num_outer, POIs) )
            all_innerObjs.append(innerObjs)
    
        print('all_POIs:', all_POIs)
        for idx_pair, pair in enumerate(all_POIs):
            print('pair:', pair)
            print('pair[1]: ', pair[1], len(pair[1]))
            for idx_POI, POI in enumerate(pair[1]):
                print('idx_POI, POI, poi_thresh:', idx_POI, POI, poi_thresh)
                if POI >= poi_thresh and pair[0] > 0.0:
                    idx_inner = ceil((idx_POI+1) / pair[0]) - 1
                    print('idx_inner, pair[0]:', idx_inner, pair[0])
                    
                    print(all_innerObjs[idx_pair][idx_inner].name)
                    obj = all_innerObjs[idx_pair][idx_inner]
    
                    OutObjs.append(obj)
                    
        return OutObjs
    
#                    out_string = WriteToFile(obj, img_file).string
#                    out.append(out_string)
#                    WriteToFile.ToFile(out, 
#                                       path.join(data_dir, 
#                                                 config['DATA']['RES_FILE']) )


                    
def WriteToFile(img_file, out_file, objs):
    if len(objs) != 0:
        obj_string = ','.join([obj.obj_string for obj in objs])
        out_string = '{"filename":"' + img_file + \
                          '","tag":[' + obj_string + ']}'

    else:
        out_string = '{"filename":"' + img_file + '","tag":[]}'

    print(out_file)
    with open(out_file, 'a') as f:
        f.write(out_string + '\n')


def ObjsPOI(innerObj, outerObj):
    w_POI = (outerObj.w + innerObj.w) \
            - ( max(innerObj.l, innerObj.r, outerObj.l, outerObj.r) 
               -min(innerObj.l, innerObj.r, outerObj.l, outerObj.r) )
    
    h_POI = (outerObj.h + innerObj.h) \
            - ( max(innerObj.t, innerObj.b, outerObj.t, outerObj.b) 
               -min(innerObj.t, innerObj.b, outerObj.t, outerObj.b) )
            
    POI = (w_POI * h_POI)*1.0 / innerObj.a if w_POI > 0 and h_POI > 0 else 0.0
    
    return POI 


def ShowImg(img):
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def SaveImg(img, save_path='./test_pic.jpg', resize_ratio=None):
    if resize_ratio is not None:
        width   = int(img.shape[0] * resize_ratio)
        height  = int(img.shape[1] * resize_ratio)

        img = cv2.resize(img, (height, width))

    cv2.imwrite(save_path, img)


def DrawBBox(objs, img, color=(0, 255, 0), width=5):
    import copy

    img_bbox = copy.deepcopy(img)

    if objs is None:
        objs = []

    for obj in objs:
        cv2.rectangle(img_bbox, (obj.l, obj.t), (obj.r, obj.b), color, width)
       
        cv2.line(img_bbox, (obj.l, obj.t), (obj.r, obj.b), color, width)
        cv2.line(img_bbox, (obj.r, obj.t), (obj.l, obj.b), color, width)

        cv2.circle(img_bbox, (obj.cx, obj.cy), 10, color, -1)
        cv2.circle(img_bbox, (int(img.shape[1]/2), int(img.shape[0]/2)), 
                   15, (0, 0, 255), -1)

        cv2.putText(img_bbox,
                    'class: ' + obj.name + '_conf: ' + str(obj.conf),
                    (obj.l, obj.t-10), 
                    cv2.FONT_HERSHEY_TRIPLEX, 
                    0.5, 
                    color, 
                    1, 
                    cv2.LINE_AA)

    return img_bbox


def AutoLabeling(img, objs, label_dict, 
                 img_path, label_path, skip_nolabel=False):
    # label_dict: {b'object1': 0, b'object2': 1, ...}
    # skip_nolabel == True : don't generate labels and images which detect nothing

    if len(objs) == 0 and skip_nolabel:
        print('skip no label.')
        pass

    else:
        height, width, channel = img.shape    
        
        with open(label_path, 'w') as f:
            if len(objs) == 0:
                cv2.imwrite(img_path, img)
        
            else:
                for obj in objs:
                    cx = obj.cx / width
                    cy = obj.cy / height
        
                    w = obj.w / width
                    h = obj.h / height
        
                    cv2.imwrite(img_path, img)
        
                    idx = label_dict[bytes(obj.name, encoding='utf8')]
 
                    f.write('{} {} {} {} {}'.format(idx, cx, cy, w, h))


# direction = "right" or "left"
def ObjFlowNum(cur_objs, pre_objs, direction, baseline):
    direction = 1 if direction=="right" else -1

    obj_pairs = []
    for cur_obj in cur_objs:
        dists = []
        for pre_obj in pre_objs:
            dist = sqrt( (cur_obj.cx - pre_obj.cx)**2 + 
                              (cur_obj.cy - pre_obj.cy)**2 )

            dists.append(dist)

        obj_pairs.append( (cur_obj, pre_objs[ dists.index(min(dists)) ]) )

    print('obj_pairs: ', len(obj_pairs), baseline)
    num_obj = 0
    for obj_pair in obj_pairs:
        print('obj_pair[0].cx: ', obj_pair[0].cx, obj_pair[1].cx)
        if (obj_pair[0].cx - baseline) * direction > 0 and \
                          (baseline - obj_pair[1].cx) * direction > 0:
            num_obj += 1

    return num_obj

# compute the relative position and Yaw between camera and sample
class CamOrient():
#    def __init__(self, objs, tgt_shape, cam_fov_deg, temp_realsize, temp_shape, label_dict, temp_objs_coord_file):
    def __init__(self, objs, tgt_shape, cam_fov_deg, temp_realsize, temp_shape, meta, temp_objs_coord_file):
        # objs: objects detected by yolo
        # tgt_shape: target (local) image shape, (height, width)
        # cam_fov_deg: camera FOV in degree (vertical, horizontal)
        # temp_realsize: real size of template image in mm or cm (height, width)
        # temp_shape: resolution of template image in pixel (height, width)
        # meta : meta from darknet
        # temp_objs_coord_file: darknet yolo labeling txt file for tracking template

        if len(objs) > 0:
            self.objs = sorted(objs, key=lambda x: x.conf, reverse=True)
        else:
            print('there is no object be detected for tracking.')

        self.tgt_shape = tgt_shape
        self.tgt_w = tgt_shape[1]
        self.tgt_h = tgt_shape[0]
        self.tgt_cx = int(tgt_shape[1]/2)
        self.tgt_cy = int(tgt_shape[0]/2)

        self.cam_fov_deg = cam_fov_deg

        # informations of template image
        self.temp_realsize = temp_realsize
        self.temp_shape = temp_shape
        self.temp_w = temp_shape[1]
        self.temp_h = temp_shape[0]
        self.temp_cx = int(temp_shape[1]/2)
        self.temp_cy = int(temp_shape[0]/2)

        self.label_dict, self.temp_objs_coord = \
            self.LoadTempInfo(meta, temp_objs_coord_file)

        if len(objs) >= 3:
            print('into self.Triangulation.')   # for_test
            self.xy_position_pixel, \
            self.position_real, \
            self.cam_height, \
            self.temp_tgt_ratio, \
            self.mm2pixel = self.Triangulation()

            self.yaw_rad, self.yaw_deg = self.CalcYaw()

        elif len(objs) == 2:
            self.xy_position_pixel, \
            self.position_real, \
            self.cam_height, \
            self.temp_tgt_ratio, \
            self.mm2pixel = self.PosMapping()

            self.yaw_rad, self.yaw_deg = self.CalcYaw()

        elif len(objs) == 1:
            self.xy_position_pixel, \
            self.position_real, \
            self.cam_height, \
            self.temp_tgt_ratio, \
            self.mm2pixel = self.PosMapping()

            self.yaw_rad = self.yaw_deg = None

        elif len(objs) == 0:
            self.xy_position_pixel = \
            self.position_real = \
            self.cam_height = \
            self.mm2pixel = None

            self.yaw_rad = self.yaw_deg = None


    def LoadTempInfo(self, meta, temp_objs_coord_file):
        label_dict = {}
        for idx in range(meta.classes):
            label_dict[meta.names[idx]] = idx

        objs_coord = []
        with open(temp_objs_coord_file, 'r') as temp_txt:
             temp_coords = temp_txt.readlines()

             for temp_coord in temp_coords:
                 coord = temp_coord.strip('\n').split()

                 objs_coord.append( (float(coord[1]), 
                                     float(coord[2]),
                                     float(coord[3]),
                                     float(coord[4])) )

        return label_dict, objs_coord

    def CalcNorm(self, a_point, b_point):
        vec = (a_point[0] - b_point[0], a_point[1] - b_point[1])

        norm = np.linalg.norm(vec)

        return norm


    def Triangulation(self):
        tri_objs = self.objs[0:3]

        for obj in tri_objs:
            print('obj: ', obj.name, obj.cx, obj.cy)

        # FOV of the camera
        fov_w_deg = self.cam_fov_deg[1]
        fov_h_deg = self.cam_fov_deg[0]
        fov_w_rad = fov_w_deg * (pi/180.0) 
        fov_h_rad = fov_h_deg * (pi/180.0)
    
        # informations of reference object
        objs_id = []
        for obj in tri_objs:
            objs_id.append(self.label_dict[bytes(obj.name, encoding='utf-8')])

        temp_objs_coord = []
        for obj_id in objs_id:
            temp_objs_coord.append(self.temp_objs_coord[obj_id])

        temp_real_w = self.temp_realsize[1]
        temp_real_h = self.temp_realsize[0]

        temp_objs_real_pos = []
        for temp_obj_coord in temp_objs_coord:
            temp_objs_real_pos.append((temp_obj_coord[0] * temp_real_w, temp_obj_coord[1] * temp_real_h))

        objs_pos = temp_objs_real_pos
        print('tri_objs: ', tri_objs)   # for_test
        print('objs_pos: ', objs_pos)   # for_test

        # distance between central point and three objects (a, b, c) in pixel 
        cen_dis = ( sqrt(pow(self.tgt_cx - tri_objs[0].cx, 2.0) + 
                         pow(self.tgt_cy - tri_objs[0].cy, 2.0)),

                    sqrt(pow(self.tgt_cx - tri_objs[1].cx, 2.0) + 
                         pow(self.tgt_cy - tri_objs[1].cy, 2.0)),

                    sqrt(pow(self.tgt_cx - tri_objs[2].cx, 2.0) + 
                         pow(self.tgt_cy - tri_objs[2].cy, 2.0)) )

        # distance between objects (A<->B, B<->C, C<->A) in pixel 
        obj_dis = ( sqrt(pow(tri_objs[0].cx - tri_objs[1].cx, 2.0) + 
                         pow(tri_objs[0].cy - tri_objs[1].cy, 2.0)),

                    sqrt(pow(tri_objs[1].cx - tri_objs[2].cx, 2.0) + 
                         pow(tri_objs[1].cy - tri_objs[2].cy, 2.0)),

                    sqrt(pow(tri_objs[2].cx - tri_objs[0].cx, 2.0) + 
                         pow(tri_objs[2].cy - tri_objs[0].cy, 2.0)) )

        # distance between objects (A<->B, B<->C, C<->A) in mm
        obj_real_dis = ( sqrt(pow(temp_objs_real_pos[0][0] - temp_objs_real_pos[1][0], 2.0) + 
                              pow(temp_objs_real_pos[0][1] - temp_objs_real_pos[1][1] , 2.0)),

                         sqrt(pow(temp_objs_real_pos[1][0] - temp_objs_real_pos[2][0], 2.0) + 
                              pow(temp_objs_real_pos[1][1] - temp_objs_real_pos[2][1] , 2.0)),

                         sqrt(pow(temp_objs_real_pos[2][0] - temp_objs_real_pos[0][0], 2.0) + 
                              pow(temp_objs_real_pos[2][1] - temp_objs_real_pos[0][1] , 2.0)) )

#        w_mm2pixel = ((tri_objs[1].cx - tri_objs[0].cx) / (objs_pos[1][0] - objs_pos[0][0]) + 
#                      (tri_objs[2].cx - tri_objs[1].cx) / (objs_pos[2][0] - objs_pos[1][0]) +
#                      (tri_objs[0].cx - tri_objs[2].cx) / (objs_pos[0][0] - objs_pos[2][0])) / 3.0
#
#        h_mm2pixel = ((tri_objs[1].cy - tri_objs[0].cy) / (objs_pos[1][1] - objs_pos[0][1]) + 
#                      (tri_objs[2].cy - tri_objs[1].cy) / (objs_pos[2][1] - objs_pos[1][1]) +
#                      (tri_objs[0].cy - tri_objs[2].cy) / (objs_pos[0][1] - objs_pos[2][1])) / 3.0

        AB_mm2pixel = self.CalcNorm((tri_objs[0].cx, tri_objs[0].cy), 
                                    (tri_objs[1].cx, tri_objs[1].cy)) / \
                      self.CalcNorm(objs_pos[0], objs_pos[1])

        BC_mm2pixel = self.CalcNorm((tri_objs[1].cx, tri_objs[1].cy), 
                                    (tri_objs[2].cx, tri_objs[2].cy)) / \
                        self.CalcNorm(objs_pos[1], objs_pos[2])

        CA_mm2pixel = self.CalcNorm((tri_objs[2].cx, tri_objs[2].cy), 
                                    (tri_objs[0].cx, tri_objs[0].cy)) / \
                        self.CalcNorm(objs_pos[2], objs_pos[0])

        w_mm2pixel = h_mm2pixel = (AB_mm2pixel+BC_mm2pixel+CA_mm2pixel) / 3.0


        # distance between central point and three objects (a, b, c) in mm
#        cen_real_dis = (cen_dis[0] * mm2pixel, cen_dis[1] * mm2pixel, cen_dis[2] )
        cen_real_dis = []
        for i in range(3):
            dis = sqrt(pow((self.tgt_cx - tri_objs[i].cx) / w_mm2pixel, 2.0) + 
                       pow((self.tgt_cy - tri_objs[i].cy) / h_mm2pixel, 2.0))

            cen_real_dis.append(dis)

        print('cen_real_dis: ', cen_real_dis)

        
        # 3x2 matrix
        A = 2 * np.array([[ objs_pos[0][0] - objs_pos[1][0], objs_pos[0][1] - objs_pos[1][1] ],
                          [ objs_pos[1][0] - objs_pos[2][0], objs_pos[1][1] - objs_pos[2][1] ],
                          [ objs_pos[2][0] - objs_pos[0][0], objs_pos[2][1] - objs_pos[0][1] ]]) 

        # 3x1 matrix
        B = np.array([[ (pow(cen_real_dis[1], 2.0) - pow(cen_real_dis[0], 2.0)) - 
                        (pow(objs_pos[1][0], 2.0) - pow(objs_pos[0][0], 2.0)) - 
                        (pow(objs_pos[1][1], 2.0) - pow(objs_pos[0][1], 2.0)) ],

                      [ (pow(cen_real_dis[2], 2.0) - pow(cen_real_dis[1], 2.0)) - 
                        (pow(objs_pos[2][0], 2.0) - pow(objs_pos[1][0], 2.0)) - 
                        (pow(objs_pos[2][1], 2.0) - pow(objs_pos[1][1], 2.0)) ],

                      [ (pow(cen_real_dis[0], 2.0) - pow(cen_real_dis[2], 2.0)) - 
                        (pow(objs_pos[0][0], 2.0) - pow(objs_pos[2][0], 2.0)) - 
                        (pow(objs_pos[0][1], 2.0) - pow(objs_pos[2][1], 2.0)) ]
                      ])


#        # 3x2 matrix
#        A = 2 * np.array([[ tri_objs[0].cx - tri_objs[1].cx, tri_objs[0].cy - tri_objs[1].cy ],
#                          [ tri_objs[1].cx - tri_objs[2].cx, tri_objs[1].cy - tri_objs[2].cy ],
#                          [ tri_objs[2].cx - tri_objs[0].cx, tri_objs[2].cy - tri_objs[0].cy ]]) 
#
#        # 3x1 matrix
#        B = np.array([[ (pow(dis[1], 2.0) - pow(dis[0], 2.0)) - 
#                        (pow(tri_objs[1].cx, 2.0) - pow(tri_objs[0].cx, 2.0)) - 
#                        (pow(tri_objs[1].cy, 2.0) - pow(tri_objs[0].cy, 2.0)) ],
#
#                      [ (pow(dis[2], 2.0) - pow(dis[1], 2.0)) - 
#                        (pow(tri_objs[2].cx, 2.0) - pow(tri_objs[1].cx, 2.0)) - 
#                        (pow(tri_objs[2].cy, 2.0) - pow(tri_objs[1].cy, 2.0)) ],
#
#                      [ (pow(dis[0], 2.0) - pow(dis[2], 2.0)) - 
#                        (pow(tri_objs[0].cx, 2.0) - pow(tri_objs[2].cx, 2.0)) - 
#                        (pow(tri_objs[0].cy, 2.0) - pow(tri_objs[2].cy, 2.0)) ]
#                      ])

        # x = (A^-1)B
        xy_position_real = np.linalg.lstsq(A, B, rcond=-1)[0].tolist()
        xy_position_real = (xy_position_real[0][0], xy_position_real[1][0])

        print('xy_position_real: ', xy_position_real)   # for_test
        print('mm2pixel: ', w_mm2pixel, h_mm2pixel)   # for_test
        xy_position_pixel = ( int(xy_position_real[0] * w_mm2pixel), 
                              int(xy_position_real[1] * h_mm2pixel) )

        cam_height = ( (self.tgt_w / (w_mm2pixel * 2.0)) / tan(fov_w_rad/2.0) + 
                       (self.tgt_h / (h_mm2pixel * 2.0)) / tan(fov_h_rad/2.0) ) / 2.0

        position_real = (xy_position_real[0], xy_position_real[1], cam_height)

        w_ratio = ( (temp_objs_coord[1][0] - temp_objs_coord[0][0]) * temp_real_w / 
                      (tri_objs[1].cx - tri_objs[0].cx) + 
                    (temp_objs_coord[2][0] - temp_objs_coord[1][0]) * temp_real_w / 
                      (tri_objs[2].cx - tri_objs[1].cx) +
                    (temp_objs_coord[0][0] - temp_objs_coord[2][0]) * temp_real_w / 
                      (tri_objs[0].cx - tri_objs[2].cx) ) / 3.0

        h_ratio = ( (temp_objs_coord[1][1] - temp_objs_coord[0][1]) * temp_real_h / 
                      (tri_objs[1].cy - tri_objs[0].cy) + 
                    (temp_objs_coord[2][1] - temp_objs_coord[1][1]) * temp_real_w / 
                      (tri_objs[2].cy - tri_objs[1].cy) +
                    (temp_objs_coord[0][1] - temp_objs_coord[2][1]) * temp_real_w / 
                      (tri_objs[0].cy - tri_objs[2].cy) ) / 3.0








        return xy_position_pixel, position_real, cam_height, \
                (w_ratio, h_ratio), (w_mm2pixel, h_mm2pixel)



    def PosMapping(self):
        obj = self.objs[0]
        
        # informations of reference object
        obj_id = self.label_dict[bytes(obj.name, encoding='utf-8')]
        obj_cx = obj.cx
        obj_cy = obj.cy
        obj_w = obj.w
        obj_h = obj.h 

        temp_obj_coord = self.temp_objs_coord[obj_id]

        # informations of template image
        temp_cx = int(self.temp_shape[1]/2)
        temp_cy = int(self.temp_shape[0]/2)
        temp_real_w = self.temp_realsize[1]
        temp_real_h = self.temp_realsize[0]

        # informations of object in template image
        temp_obj_cx = int(temp_obj_coord[0] * self.temp_w)
        temp_obj_cy = int(temp_obj_coord[1] * self.temp_h)
        temp_obj_w = int(temp_obj_coord[2] * self.temp_w) 
        temp_obj_h = int(temp_obj_coord[3] * self.temp_h) 
        temp_obj_real_w = temp_obj_coord[2] * temp_real_w 
        temp_obj_real_h = temp_obj_coord[3] * temp_real_h 

        # mm to pixel
        w_mm2pixel = obj_w / temp_obj_real_w 
        h_mm2pixel = obj_h / temp_obj_real_h

        # ratio between target image and template image
        w_ratio = temp_obj_w / obj_w
        h_ratio = temp_obj_h / obj_h

        # FOV of the camera
        fov_w_deg = self.cam_fov_deg[1]
        fov_h_deg = self.cam_fov_deg[0]
        fov_w_rad = fov_w_deg * (pi/180.0) 
        fov_h_rad = fov_h_deg * (pi/180.0)

        # informations of target image
        tgt_w = self.tgt_shape[1] 
        tgt_h = self.tgt_shape[0]
        tgt_cx = int(tgt_w/2)
        tgt_cy = int(tgt_h/2)
        tgt_real_w = temp_obj_real_w * (tgt_w / obj_w)
        tgt_real_h = temp_obj_real_h * (tgt_h / obj_h)

        # relative position between the center of reference object and the target image
        shift_vec = (tgt_cx-obj_cx, tgt_cy-obj_cy)
#        print()
#        print('obj_id: ', obj_id, 'conf: ', obj.conf)
#        print('shift_vec: ', shift_vec)

        hypotenuse_w = (tgt_real_w / sin(fov_w_rad)) * sin((pi-fov_w_rad)/2.0)
        hypotenuse_h = (tgt_real_h / sin(fov_h_rad)) * sin((pi-fov_h_rad)/2.0)
    
        cam_height = sqrt( pow(hypotenuse_w, 2.0) - pow(tgt_real_w/2.0, 2.0) )
#        cam_height = sqrt( pow(hypotenuse_h, 2.0) - pow(tgt_real_h/2.0, 2.0) )

        xy_position_pixel = ( temp_obj_cx + shift_vec[0] * w_ratio,
                              temp_obj_cy + shift_vec[1] * h_ratio )

        position_real = ( (xy_position_pixel[0] / self.temp_w) * temp_real_w,
                          (xy_position_pixel[1] / self.temp_h) * temp_real_h,
                          cam_height )

        return xy_position_pixel, position_real, cam_height, \
                (w_ratio, h_ratio), (w_mm2pixel, h_mm2pixel)


    def CalcYaw(self) -> (float, float):
        # informations of reference objects
        obj1_id = self.label_dict[bytes(self.objs[0].name, encoding='utf-8')]
        obj2_id = self.label_dict[bytes(self.objs[1].name, encoding='utf-8')]

        obj1_cx = self.objs[0].cx
        obj1_cy = self.objs[0].cy

        obj2_cx = self.objs[1].cx
        obj2_cy = self.objs[1].cy

        print('obj_name: ', obj1_id, self.objs[0].name, obj2_id, self.objs[1].name)
        print('obj_c: ', obj1_cx, obj1_cy, obj2_cx, obj2_cy)

        temp_obj1_cx = int(self.temp_objs_coord[obj1_id][0] * self.temp_w)
        temp_obj1_cy = int(self.temp_objs_coord[obj1_id][1] * self.temp_h)

        temp_obj2_cx = int(self.temp_objs_coord[obj2_id][0] * self.temp_w)
        temp_obj2_cy = int(self.temp_objs_coord[obj2_id][1] * self.temp_h)

        print('temp_obj_c: ', temp_obj1_cx, temp_obj1_cy, temp_obj2_cx, temp_obj2_cy)

        tgt_vec = np.array([(obj2_cx - obj1_cx) * self.temp_tgt_ratio[0], 
                            (obj2_cy - obj1_cy) * self.temp_tgt_ratio[1]])

        temp_vec = np.array([temp_obj2_cx - temp_obj1_cx, 
                             temp_obj2_cy - temp_obj1_cy])

        print('yaw_vec: ', tgt_vec, temp_vec)

        tgt_vec_norm = np.linalg.norm(tgt_vec)
        temp_vec_norm = np.linalg.norm(temp_vec)

        if tgt_vec_norm * temp_vec_norm == 0.0:
            yaw_rad = yaw_deg = None

            return yaw_rad, yaw_deg

        else:
            yaw_rad = np.arccos( np.dot(tgt_vec, temp_vec) / (tgt_vec_norm * temp_vec_norm))
            rotation = np.cross(temp_vec, tgt_vec)

            if rotation < 0: yaw_rad *= -1

            yaw_deg = yaw_rad * 180.0 / pi
       
        print('yaw, norm: ', yaw_rad, yaw_deg, tgt_vec_norm, temp_vec_norm)

        return float(yaw_rad), float(yaw_deg)



