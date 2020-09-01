#!/usr/bin/env python3

import YoloObj
import DarknetFunc as DFUNC

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


if __name__ == '__main__':
    import cv2

    darknet_cfg = '/mnt/sdb1/work/ADAT/template_matching/yolotrain7/cfg/yolov3_test.cfg'
    darknet_weights = '/mnt/sdb1/work/ADAT/template_matching/yolotrain7/backup/yolov3_best.weights'
    darknet_data = '/mnt/sdb1/work/ADAT/template_matching/yolotrain7/cfg/adat.data'
    img_file = '/mnt/sdb1/work/ADAT/template_matching/yolotrain7/yolomark/template_cut.jpg'

    img_np = cv2.imread(img_file)

    DFUNC.set_gpu(0)
    net = DFUNC.load_net(bytes(darknet_cfg, 'utf-8'), bytes(darknet_weights, 'utf-8'), 0)
    meta = DFUNC.load_meta(bytes(darknet_data, 'utf-8'))

    objs = ImgDetect(net, meta, img_np, thresh=0.25) 

    print(objs, len(objs))

    img_np = YoloObj.DrawBBox(objs, img_np)
    YoloObj.ShowImg(img_np)
    

