#coding=utf-8
import json

import cv2
import os
import xml.dom
import xml.dom.minidom
import numpy as np
import torch
from detectron2.config import get_cfg
from detectron2.engine.defaults import DefaultPredictor
from detectron2.data import MetadataCatalog
from detectron2.engine import default_setup
from detectron2.structures import Boxes
from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)

class Model():
    
    def __init__(self,cfg_path,weights_path):
    
        #nms_same_class 0.3  ----  *.yaml/TEST:NMS:0.3 中设置 defalut 0.5
        
        self.gpu_id = 0               #gpu_id default 0
        
        self.score_thresh = 0.4       #score > score_thresh  default 0.3  
        
        self.per_class_thresh = False         #score > class_score_thresh
        self.autotruck_score_thresh = 0.6
        self.forklift_score_thresh = 0.65
        self.digger_score_thresh = 0.65
        self.car_score_thresh = 0.45
        self.bus_score_thresh = 0.0
        self.tanker_score_thresh = 0.55
        self.person_score_thresh = 0.35
        self.minitruck_score_thresh = 0.0
        self.minibus_score_thresh = 0.59
        
        self.class_nms_thresh = 0.85   #nms_between_classes  IOU > class_nms_thresh    default 0.9 
        
        ###############
        self.class_name = [k["name"] for k in COCO_CATEGORIES if k["isthing"] == 1]
        self.cpu_device = torch.device("cpu")
        cfg = get_cfg()
        cfg.merge_from_file(cfg_path)

        num_gpus = torch.cuda.device_count()
        gpuid = 0
        cfg.MODEL.DEVICE = "cuda:{}".format(gpuid) if num_gpus > 0 else "cpu"
        self.predictor = DefaultPredictor(cfg)
        
        self.crossAreaRatios = 0.1        
        self.PolyDebug = False
        if self.PolyDebug == True:
            self.img=np.zeros((1080,1920,3)) 
        
        print ("model is ok")

    def predict(self,im, x_position, y_position):

        #class_str_list = []        
        
        if len(x_position) > 2 :            
            position_num = len(x_position)
            
        if (self.PolyDebug == True and position_num > 2):
            ll=[]
            for i in range(position_num):
                ll.append([x_position[i],y_position[i]])
            cv2.fillConvexPoly(self.img, np.array(ll, np.int32), (255,255,0))        
        
        data_list = []
        
        predictions = self.predictor(im)   #detectron2/demo/predictor.py
        instances = predictions["instances"].to(self.cpu_device)
        boxes = instances.pred_boxes if instances.has("pred_boxes") else None  #<class 'detectron2.structures.boxes.Boxes'>
        
        scores = instances.scores if instances.has("scores") else None  #<class 'torch.Tensor'>
        classes = instances.pred_classes if instances.has("pred_classes") else None  ##<class 'torch.Tensor'>
        
        classes_lst = []
        scores_lst = []
        if scores is not None:
            scores_lst = [round(float(s),2) for s in scores] #["{:.0f}".format(s * 100) for s in scores] #[round(float(s),2) for s in scores]
        if classes is not None:
            classes_lst = [int(c) for c in classes]
                
        num_instances = None
        if boxes is not None:
            boxes = self._convert_boxes(boxes)
            num_instances = len(boxes)
       
        areas = None
        if boxes is not None:
            areas = np.prod(boxes[:, 2:] - boxes[:, :2], axis=1)

        if areas is not None:
            sorted_inds = np.argsort(-areas).tolist()
            # Re-order overlapped instances in descending order.
            boxes = boxes[sorted_inds] if boxes is not None else None
            scores_lst = [scores_lst[k] for k in sorted_inds] if scores_lst is not None else None 
            classes_lst = [classes_lst[k] for k in sorted_inds] if classes_lst is not None else None
           
            #print (scores_lst)
            '''for i in range(len(boxes)):
                bbox = boxes[i]
                score = scores_lst[i]
                classe = classes_lst[i]
                print (i,bbox,score,self.class_name[classe])
                if score>0.4:
                    single_data = {"cls":" ","score":float('%.2f' % score),"bbox":{"xmin":int(bbox[0]),"ymin":int(bbox[1]),"xmax":int(bbox[2]),"ymax":int(bbox[3])}}
                    data_list.append(single_data)'''
        else:
            return data_list
        
        #no nms between classes
        '''im1 = cv2.cvtColor(im,cv2.COLOR_RGB2BGR)
        result1= im1.copy()
        for i in sorted_inds:
            
            bbox = boxes[i, :4]
            score = boxes[i, -1]
            if score < self.score_thresh:
                continue
            #get class-str
            class_str = self.get_class_string(classes[i], score, self.dummy_coco_dataset)            
            cv2.rectangle(result1,(int(bbox[0]),int(bbox[1])),(int(bbox[2]),int(bbox[3])),(255,255,0),1)
            font = cv2.FONT_HERSHEY_SIMPLEX
            ((txt_w, txt_h), _) = cv2.getTextSize(class_str, font, 0.35, 1)            
            txt_tl = int(bbox[0]), int(bbox[1]) - int(0.3 * txt_h)
            cv2.putText(result1, class_str, txt_tl, font, 0.35, (218, 227, 218), lineType=cv2.LINE_AA)
            txt_tl = int(bbox[0])+txt_w, int(bbox[1]) - int(0.3 * txt_h)
            cv2.putText(result1, ('%.2f' % score), txt_tl, font, 0.35, (218, 227, 218), lineType=cv2.LINE_AA)
        cv2.imwrite("test1.jpg", result1)'''
        
        #nms between classes
        #im2 = cv2.cvtColor(im,cv2.COLOR_RGB2BGR)
        #result2= im2.copy()        
        if (len(sorted_inds) > 0):        
            nmsIndex = self.nms_between_classes(boxes, scores_lst, self.class_nms_thresh)  #阈值为0.9，阈值越大，过滤的越少 
            for i in range(len(nmsIndex)):                
                bbox = boxes[nmsIndex[i], :4]
                score = float(scores_lst[nmsIndex[i]])
                class_str = self.class_name[classes_lst[nmsIndex[i]]]
                if score < self.score_thresh:
                    continue

                #score thresd per class
                if self.per_class_thresh:
                    if 'autotruck' == class_str and score < self.autotruck_score_thresh:
                        continue
                    if 'forklift'  == class_str and score < self.forklift_score_thresh:
                        continue
                    if 'digger'    == class_str and score < self.digger_score_thresh:
                        continue
                    if 'car'       == class_str and score < self.car_score_thresh:
                        continue
                    if 'bus'       == class_str and score < self.bus_score_thresh:
                        continue
                    if 'tanker'    == class_str and score < self.tanker_score_thresh:
                        continue
                    if 'person'    == class_str and score < self.person_score_thresh:
                        continue
                    if 'minitruck' == class_str and score < self.minitruck_score_thresh:
                        continue
                    if 'minibus'   == class_str and score < self.minibus_score_thresh:
                        continue                
                
                if len(x_position) > 2 :
                    bbox_x = [int(bbox[0]), int(bbox[2]), int(bbox[2]), int(bbox[0])]
                    bbox_y = [int(bbox[1]), int(bbox[1]), int(bbox[3]), int(bbox[3])]
                    if self.IsFilterByElectronicFence(bbox_x, bbox_y, x_position, y_position):
                        continue
                        
                single_data = {"cls":class_str,"score":float('%.2f' % score),"bbox":{"xmin":int(bbox[0]),"ymin":int(bbox[1]),"xmax":int(bbox[2]),"ymax":int(bbox[3])}}
                data_list.append(single_data)        
        
                '''cv2.rectangle(result2,(int(bbox[0]),int(bbox[1])),(int(bbox[2]),int(bbox[3])),(255,255,0),1)
                font = cv2.FONT_HERSHEY_SIMPLEX
                ((txt_w, txt_h), _) = cv2.getTextSize(class_str, font, 0.55, 1)            
                txt_tl = int(bbox[0]), int(bbox[1]) - int(0.3 * txt_h)
                cv2.putText(result2, class_str, txt_tl, font, 0.55, (218, 227, 218), lineType=cv2.LINE_AA)
                txt_tl = int(bbox[0])+txt_w, int(bbox[1]) - int(0.3 * txt_h)
                cv2.putText(result2, ('%.2f' % score), txt_tl, font, 0.35, (218, 227, 218), lineType=cv2.LINE_AA)'''
        #cv2.imwrite("test2.jpg", result2)
        
        #construcrion - data_list
        if self.PolyDebug == True:
            cv2.imwrite("1.jpg", self.img)
        return data_list
        
    def _convert_boxes(self, boxes):
        """
        Convert different format of boxes to a Nx4 array.
        """
        if isinstance(boxes, Boxes):
            return boxes.tensor.numpy()
        else:
            return np.asarray(boxes)   
            
    def convert_from_cls_format(self,cls_boxes, cls_segms, cls_keyps):
        """Convert from the class boxes/segms/keyps format generated by the testing
        code.
        """
        box_list = [b for b in cls_boxes if len(b) > 0]
        if len(box_list) > 0:
            boxes = np.concatenate(box_list)
        else:
            boxes = None
        if cls_segms is not None:
            segms = [s for slist in cls_segms for s in slist]
        else:
            segms = None
        if cls_keyps is not None:
            keyps = [k for klist in cls_keyps for k in klist]
        else:
            keyps = None
        classes = []
        for j in range(len(cls_boxes)):
            classes += [j] * len(cls_boxes[j])
        return boxes, segms, keyps, classes
        
    def get_class_string(self,class_index, score, dataset):
        class_text = dataset.classes[class_index] if dataset is not None else \
            'id{:d}'.format(class_index)
        #return class_text + ' {:0.2f}'.format(score).lstrip('0')
        return class_text
    def nms_between_classes(self,boxes, scores_lst,threshold):
        if boxes.size==0:
            return np.empty((0,3))
        x1 = boxes[:,0]
        y1 = boxes[:,1]
        x2 = boxes[:,2]
        y2 = boxes[:,3]
        s = scores_lst
        area = (x2-x1+1) * (y2-y1+1)
        I = np.argsort(s)
        pick = np.zeros_like(s, dtype=np.int16)
        counter = 0
        while I.size>0:
            i = I[-1]
            pick[counter] = i
            counter += 1
            idx = I[0:-1]
            xx1 = np.maximum(x1[i], x1[idx])
            yy1 = np.maximum(y1[i], y1[idx])
            xx2 = np.minimum(x2[i], x2[idx])
            yy2 = np.minimum(y2[i], y2[idx])
            w = np.maximum(0.0, xx2-xx1+1)
            h = np.maximum(0.0, yy2-yy1+1)
            inter = w * h        
            o = inter / (area[i] + area[idx] - inter)
            I = I[np.where(o<=threshold)]
        pick = pick[0:counter]  #返回nms后的索引
        return pick
        
    def IsFilterByElectronicFence(self,bbox_x, bbox_y, x_position, y_position): 
        #电子围栏的边界
        ex_position_max = max(x_position)
        ex_position_min = min(x_position)
        ey_position_max = max(y_position)
        ey_position_min = min(y_position)
        e_position_num = len(x_position)
        
        if self.PolyDebug == True:
            self.drawInfo(bbox_x, bbox_y)
    
        x_position_max = max(bbox_x)
        x_position_min = min(bbox_x)
        y_position_max = max(bbox_y)
        y_position_min = min(bbox_y)        
               
        #在多边形的外围，需要过滤掉
        if ((y_position_max < ey_position_min)or(y_position_min > ey_position_max)
            or(x_position_max < ex_position_min)or(x_position_min > ex_position_max)):
            print("box min max out of poly:", bbox_x, bbox_y)
            return True
        
        #依次判断矩形框的每个点是否在多边形内
        Inpoly_rst=[0,0,0,0]
        for i in range(4):
            Inpoly_rst[i] = self.IsPtInPoly(bbox_x[i],bbox_y[i],x_position, y_position)
        
        '''CenterPt_rst = self.IsPtInPoly((bbox_x[0]+bbox_x[1])/2,(bbox_y[0]+bbox_y[3])/2)
        
        #四个顶点全部在多边形外，则在多边形外部，需要过滤掉
        if Inpoly_rst==[0,0,0,0]:
            if CenterPt_rst==0:
                print("box out of poly, center out of poly:", bbox_x, bbox_y)
                return True
            else:
                PolyArea = self.calcPolyArea(self.x_position,self.y_position)
                crossAreaRatios = PolyArea/((bbox_x[1]-bbox_x[0]) *(bbox_y[3] - bbox_y[0])+0.001)
                print("box out of poly, center in poly:", bbox_x, bbox_y)
                print("crossAreaRatios:", crossAreaRatios)
                if (crossAreaRatios < self.crossAreaRatios):
                    
                    return True
                else:
                    return False '''
                
        
        #四个顶点全部在多边形内，则在多边形内部，不需要过滤掉
        if Inpoly_rst==[1,1,1,1]:
            print("box in poly:", bbox_x, bbox_y)
            return False 
        
        if (self.InPolyAreaRatios(bbox_x,bbox_y,Inpoly_rst,x_position, y_position) > self.crossAreaRatios):
            return False
        else:               
            return True         
    
    def IsPtInPoly(self,x,y,x_position, y_position):
    
        #电子围栏的边界
        ex_position_max = max(x_position)
        ex_position_min = min(x_position)
        ey_position_max = max(y_position)
        ey_position_min = min(y_position)
        e_position_num = len(x_position)
        
        if ((y < ey_position_min)or(y > ey_position_max)
            or(x < ex_position_min)or(x > ex_position_max)):
            return 0
        
        rst = 0        
        j = e_position_num -1
        for i in range(e_position_num):
            if ((y_position[i] > y) != (y_position[j] > y)) and (x < ((x_position[j]-x_position[i])*(y-y_position[i])/(y_position[j]-y_position[i]++0.00001) + x_position[i])):
                if rst == 0:
                    rst = 1
                else:
                    rst = 0
            j = i             
        return rst
     
    def calcPolyArea(self,x_pos,y_pos):
        PolyArea = 0
        
        if len(x_pos) < 3:
            return 0           
        
        j = len(x_pos) -1
        for i in range(len(x_pos)):            
            PolyArea = PolyArea +  (x_pos[i]*y_pos[j] - x_pos[j]*y_pos[i])
            j = i
            
        PolyArea = PolyArea/2.0 
        
        return abs(PolyArea)

     
    def InPolyAreaRatios(self,bbox_x, bbox_y, Inpoly_rst,x_position, y_position): 
    
        e_position_num = len(x_position)
        
        #计算矩形和多边形的交点,该函数的矩形框必然是和多边形相交的
        print("box out or cross Poly:", bbox_x,bbox_y)
        box_x =[bbox_x[0],bbox_x[1]]
        box_y =[bbox_y[0],bbox_y[3]]
        
        cross_info_lst=[]
        for x in box_x:        
            j = e_position_num -1        
            for i in range(e_position_num):
                if ((x_position[i] > x) != (x_position[j] > x)):                    
                    cross_y = int((y_position[j]-y_position[i])*(x-x_position[i])/(x_position[j]-x_position[i]+0.00001)+ y_position[i])
                    if ((cross_y < box_y[1]) and (cross_y > box_y[0])):
                        cross_info = [x,cross_y]
                        cross_info_lst.append(cross_info)
                j=i
        
        for y in box_y:        
            j = e_position_num -1        
            for i in range(e_position_num):
                if ((y_position[i] > y) != (y_position[j] > y)):
                    #print("y cross:",self.y_position[i],self.y_position[j],y)
                    cross_x = int((x_position[j]-x_position[i])*(y-y_position[i])/(y_position[j]-y_position[i]+0.00001) + x_position[i])
                    #print(self.x_position[j],self.x_position[i],self.y_position[j],self.y_position[i])
                    #print("x cross:",cross_x,box_x[0], box_x[1])
                    if ((cross_x < box_x[1]) and (cross_x > box_x[0])):
                        cross_info = [cross_x, y]
                        cross_info_lst.append(cross_info)
                j=i
        
        print("cross_info_lst",cross_info_lst)
        x_pos,y_pos = self.SortCrossPtAndBoxPt(cross_info_lst,bbox_x,bbox_y,Inpoly_rst,x_position, y_position) 
        print("SortCrossPtAndBoxPt x_pos after:", x_pos)
        print("SortCrossPtAndBoxPt y_pos after", y_pos)  
        
        PolyArea = self.calcPolyArea(x_pos,y_pos)
        crossAreaRatios = PolyArea/((bbox_x[1]-bbox_x[0]) *(bbox_y[3] - bbox_y[0])+0.00001)
        print("crossAreaRatios:", crossAreaRatios)      
        return crossAreaRatios 
        
        
        
    def SortCrossPtAndBoxPt(self, cross_info_lst,bbox_x,bbox_y,Inpoly_rst,x_position, y_position):  
        e_position_num = len(x_position)
        
        x_pos=[]
        y_pos=[]
        
        #加入在多边形内的矩形框的顶点
        for i in range(4):
            if Inpoly_rst[i] == 1:
                x_pos.append(bbox_x[i])
                y_pos.append(bbox_y[i])        
        
        #加入矩形框和多边形的交点
        for cross_pt in cross_info_lst:                     
            x_pos.append(cross_pt[0])
            y_pos.append(cross_pt[1])             
        
        #加入在矩形框内的多边形的顶点
        for i in range(e_position_num):            
            if (((x_position[i]< bbox_x[1]) and (x_position[i]> bbox_x[0])) 
                and((y_position[i]< bbox_y[3]) and (y_position[i]> bbox_y[0]))):                 
                x_pos.append(x_position[i])
                y_pos.append(y_position[i])
        
        print("SortCrossPtAndBoxPt x_pos before:", x_pos)
        print("SortCrossPtAndBoxPt y_pos before", y_pos)        
        
        return self.ClockwiseSortPoints(x_pos,y_pos)
        
    def ClockwiseSortPoints(self, x_pos,y_pos):
        num = len(x_pos)
        if num < 3:
            return  x_pos,y_pos
        acc_x = 0
        acc_y = 0  
        for i in range(num):
            acc_x = acc_x + x_pos[i]
            acc_y = acc_y + y_pos[i]
        centerO = [acc_x/num, acc_y/num]
        
        for i in range(num):
            for j in range(num-i-1):
                if self.PointCmp([x_pos[j],y_pos[j]],[x_pos[j+1],y_pos[j+1]], centerO):
                    pt_tmp= [x_pos[j],y_pos[j]]
                    x_pos[j] = x_pos[j+1]
                    y_pos[j] = y_pos[j+1]
                    x_pos[j+1] = pt_tmp[0]
                    y_pos[j+1] = pt_tmp[1]  
        return  x_pos,y_pos
    
    def PointCmp(self,pt1,pt2,centerO):
        #向量叉乘
        det = (pt1[0] - centerO[0])*(pt2[1] - centerO[1])- (pt1[1] - centerO[1])*(pt2[0] - centerO[0])
        if det < 0:
            return True
        if det > 0:
            return False
            
        #向量OA和向量OB共线，以距离判断大小
        d1 = pow((pt1[0] - centerO[0]),2)+pow((pt1[1] - centerO[1]),2)
        d2 = pow((pt2[0] - centerO[0]),2)+pow((pt2[1] - centerO[1]),2)
        
        return d1 > d2

    def drawInfo(self,bbox_x, bbox_y):
        cv2.rectangle(self.img,(bbox_x[0],bbox_y[0]),(bbox_x[1],bbox_y[3]),(255,0,255),2)
        
    
        
        
