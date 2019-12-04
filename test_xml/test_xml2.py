#coding=utf-8
#__author__ = 'zj 2019-10-23'

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

WRITE_XML = True
PLOT      = True
    
class Model():
    
    def __init__(self,cfg_path,weights_path):
    
        #nms_same_class 0.3  ----  *.yaml/TEST:NMS:0.3 中设置 defalut 0.5
        
        self.gpu_id = 0              #gpu_id default 0
        
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
        
        self.class_name = [k["name"] for k in COCO_CATEGORIES if k["isthing"] == 1]
        #print (self.class_name)
        
        self.class_nms_thresh = 0.85   #nms_between_classes  IOU > class_nms_thresh    default 0.9 
        #merge_cfg_from_file(cfg_path)
        self.cpu_device = torch.device("cpu")
        #dataName = ("coco_2017_oil_train",)
        #self.metadata = MetadataCatalog.get(None)
        cfg = get_cfg()
        cfg.merge_from_file(cfg_path)

        num_gpus = torch.cuda.device_count()
        gpuid = 0
        cfg.MODEL.DEVICE = "cuda:{}".format(gpuid) if num_gpus > 0 else "cpu"
        self.predictor = DefaultPredictor(cfg)
        
        #self.instance_mode = ColorMode.IMAGE
        
        #self.model = infer_engine.initialize_model_from_cfg(weights_path,self.gpu_id)
        #self.dummy_coco_dataset = dummy_datasets.get_steal_oil_class14_dataset()
        print ("model is ok")

    def predict(self,im):

        #class_str_list = []
        data_list = []
        
        '''with c2_utils.NamedCudaScope(self.gpu_id):
            cls_boxes, cls_segms, cls_keyps = infer_engine.im_detect_all(self.model, im, None, None
            )'''
        predictions = self.predictor(im)   #detectron2/demo/predictor.py
        instances = predictions["instances"].to(self.cpu_device)
        boxes = instances.pred_boxes if instances.has("pred_boxes") else None  #<class 'detectron2.structures.boxes.Boxes'>
        #<class 'detectron2.structures.boxes.Boxes'>
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
        '''if labels is not None:
            assert len(labels) == num_instances'''
        
        areas = None
        if boxes is not None:
            areas = np.prod(boxes[:, 2:] - boxes[:, :2], axis=1)

        if areas is not None:
            sorted_inds = np.argsort(-areas).tolist()
            # Re-order overlapped instances in descending order.
            boxes = boxes[sorted_inds] if boxes is not None else None
            scores_lst = [scores_lst[k] for k in sorted_inds] if scores_lst is not None else None 
            classes_lst = [classes_lst[k] for k in sorted_inds] if classes_lst is not None else None
            #labels = [labels[k] for k in sorted_inds] if labels is not None else None
            
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

        #nms between classes
        #im2 = cv2.cvtColor(im,cv2.COLOR_RGB2BGR)
        #result2= im2.copy()
        if (len(sorted_inds) > 0):        
            nmsIndex = self.nms_between_classes(boxes, scores_lst, self.class_nms_thresh)  #阈值为0.9，阈值越大，过滤的越少 
            for i in range(len(nmsIndex)):                
                bbox = boxes[nmsIndex[i], :4]
                score = float(scores_lst[nmsIndex[i]])
                #get class-str
                class_str = self.class_name[classes_lst[nmsIndex[i]]]#self.get_class_string(classes[nmsIndex[i]], score, self.dummy_coco_dataset)
                if score < self.score_thresh:
                    continue
                #print (i, bbox, score, class_str, " nms")
                
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
                
                single_data = {"cls":class_str,"score":float('%.2f' % score),"bbox":{"xmin":int(bbox[0]),"ymin":int(bbox[1]),"xmax":int(bbox[2]),"ymax":int(bbox[3])}}
                data_list.append(single_data)        
        
        #construcrion - data_list
        return data_list
    
    def _convert_boxes(self, boxes):
        """
        Convert different format of boxes to a Nx4 array.
        """
        if isinstance(boxes, Boxes):
            return boxes.tensor.numpy()
        else:
            return np.asarray(boxes)
   
    def _create_text_labels(self, classes, scores, class_names):
        """
        Args:
            classes (list[int] or None):
            scores (list[float] or None):
            class_names (list[str] or None):

        Returns:
            list[str] or None
        """
        labels = None
        if class_names is not None and len(class_names) > 1:
            labels = [class_names[i] for i in classes]
        if scores is not None:
            if labels is None:
                labels = ["{:.0f}%".format(s * 100) for s in scores]
            else:
                labels = ["{} {:.0f}%".format(l, s * 100) for l, s in zip(labels, scores)]
        return labels
    
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
    def nms_between_classes(self,boxes, scores_lst, threshold):
        if boxes.size==0:
            return np.empty((0,3))
        x1 = boxes[:,0]
        y1 = boxes[:,1]
        x2 = boxes[:,2]
        y2 = boxes[:,3]
        s = scores_lst #scores #boxes[:,4]
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

class Xml():
    def __init__(self):
        self.INDENT= ' '*4
        self.NEW_LINE= '\n'
        self.FOLDER_NODE= 'VOC2010'
        self.ROOT_NODE= 'annotation'
        self.DATABASE_NAME= 'VOC2010'
        self.ANNOTATION= 'PASCALVOC2010'
        self.AUTHOR= 'HHJ'
        self.SEGMENTED= '0'
        self.DIFFICULT= '0'
        self.TRUNCATED= '0'
        self.OCCLUDED = '0'
        self.POSE= 'Unspecified'

    def xml(self,outpath,outname,predict_datalist,img_size):
        my_dom = xml.dom.getDOMImplementation()
        doc = my_dom.createDocument(None,self.ROOT_NODE,None)
        # 获得根节�?
        root_node = doc.documentElement
        # folder节点
        self.createChildNode(doc, 'folder',self.FOLDER_NODE, root_node)
        # filename节点
        self.createChildNode(doc, 'filename', outname.rsplit('.', 1)[0]+'.jpg',root_node)
        # source节点
        source_node = doc.createElement('source')
        # source的子节点
        self.createChildNode(doc, 'database',self.DATABASE_NAME, source_node)
        self.createChildNode(doc, 'annotation',self.ANNOTATION, source_node)
        self.createChildNode(doc, 'image','flickr', source_node)
        self.createChildNode(doc, 'flickrid','NULL', source_node)
        root_node.appendChild(source_node)
        # owner节点
        owner_node = doc.createElement('owner')
        # owner的子节点
        self.createChildNode(doc, 'flickrid','NULL', owner_node)
        self.createChildNode(doc, 'name',self.AUTHOR, owner_node)
        root_node.appendChild(owner_node)
        # size节点
        size_node = doc.createElement('size')
        self.createChildNode(doc, 'width',str(img_size[1]), size_node)
        self.createChildNode(doc, 'height',str(img_size[0]), size_node)
        self.createChildNode(doc, 'depth',str(img_size[2]), size_node)
        root_node.appendChild(size_node)
        # segmented节点
        self.createChildNode(doc, 'segmented',self.SEGMENTED, root_node)
        #添加图片的 类别
        for j in range(len(predict_datalist)):
            singledata = predict_datalist[j]
            object_node = self.createObjectNode(doc, singledata)
            root_node.appendChild(object_node)
        # # 写入文件
        #
        self.writeXMLFile(doc,outpath,outname)
            
    def createElementNode(self,doc,tag, attr):  # 创建一个元素节点
        element_node = doc.createElement(tag)
        # 创建一个文本节点
        text_node = doc.createTextNode(attr)
        # 将文本节点作为元素节点的子节点
        element_node.appendChild(text_node)
        return element_node

# 封装添加一个子节点的过
    def createChildNode(self,doc,tag, attr,parent_node):
        child_node = self.createElementNode(doc, tag, attr)
        parent_node.appendChild(child_node)

    # object节点比较特殊
    def createObjectNode(self,doc,attrs):
        object_node = doc.createElement('object')
        self.createChildNode(doc, 'name', attrs['cls'],object_node)
        self.createChildNode(doc, 'pose',self.POSE, object_node)
        self.createChildNode(doc, 'truncated',self.TRUNCATED, object_node)
        self.createChildNode(doc, 'difficult',self.DIFFICULT, object_node)
        self.createChildNode(doc, 'occluded',self.OCCLUDED, object_node)
        bndbox_node = doc.createElement('bndbox')
        self.createChildNode(doc, 'xmin', str(int(attrs['bbox']['xmin'])),bndbox_node)
        self.createChildNode(doc, 'ymin', str(int(attrs['bbox']['ymin'])),bndbox_node)
        self.createChildNode(doc, 'xmax', str(int(attrs['bbox']['xmax'])),bndbox_node)
        self.createChildNode(doc, 'ymax', str(int(attrs['bbox']['ymax'])),bndbox_node)
        object_node.appendChild(bndbox_node)
        return object_node

    # 将documentElement写入XML文件�?
    def writeXMLFile(self,doc,outpath,filename):
        tmpfile =open(os.path.join(outpath,filename),'w')
        doc.writexml(tmpfile, addindent=self.INDENT,newl = '\n',encoding = 'utf-8')
        tmpfile.close()

def main(cfg_path,weights_path,input_imagespath,output_image,output_xmlpath):
    #check path
    if not os.path.exists(input_imagespath):
        print ("input_imagespath is not exist!!!")
        return 
    if not os.path.exists(output_image):
        os.makedirs(output_image)
    if not os.path.exists(output_xmlpath):
        os.makedirs(output_xmlpath)
    #init
    mm = Model(cfg_path,weights_path)
    xml = Xml()
    
    i=0
    num_no_target = 0
    files= os.listdir(input_imagespath)
    for file in files:
        if file.endswith('.jpg'):
            i = i + 1
            print(i)
            image = os.path.join(input_imagespath,file)
            print(image)
            img_np = cv2.imread(image)
            if img_np is None:
                continue
            img_size = img_np.shape
            if len(img_size) is not 3:
                print ('{} is not the right size!!!'.format(file))
                continue
            #predict
            datalist = []
            datalist = mm.predict(img_np)
            if len(datalist) < 1:
                print ('{} has not target!!!'.format(file))
                num_no_target = num_no_target + 1
                '''
                if 1: #save no target
                    output_image_temp = os.path.join(output_image,"NoTarget")
                    if not os.path.exists(output_image_temp):
                        os.makedirs(output_image_temp) 
                    cv2.imwrite(os.path.join(output_image_temp,file), img_np)
                    continue'''
                
            #write xml
            if WRITE_XML:
                xmlfile = file.rsplit('.', 1)[0] + '.xml'
                xml.xml(output_xmlpath,xmlfile,datalist,img_size)
            #plot
            if PLOT:
                re_image = os.path.join(output_image,file)
                if not os.path.exists(re_image):
                    re_image = os.path.join(input_imagespath,file)
                    
                re_img_np = cv2.imread(re_image)
                for j in range(len(datalist)):
                    singledata = {}
                    boxdict = {}
                    singledata = datalist[j]
                    boxdict = singledata['bbox']
                    xmin = boxdict['xmin']
                    ymin = boxdict['ymin']
                    xmax = boxdict['xmax']
                    ymax = boxdict['ymax']
                    cv2.rectangle(re_img_np, (xmin,ymin), (xmax,ymax),(0,0,255),2)
                    #cv2.rectangle(re_img_np, (xmin-2,ymin-2), (xmax+2,ymax+2),(0,255,0),2)
                    #cv2.rectangle(re_img_np, (xmin-5,ymin-5), (xmax+5,ymax+5),(255,0,0),2)
                    font= cv2.FONT_HERSHEY_SIMPLEX
                    strname = singledata['cls']
                    strscore = singledata['score']
                    #print (type(strscore))
                    print (strscore, strname)
                    #cv2.putText(re_img_np, strname, (xmin,ymin-5), font, 1,(0,0,255),2)
                    #cv2.putText(re_img_np, str(strscore), (xmin,ymin), font, 1,(0,0,255),2)
                    
                    cv2.putText(re_img_np, strname + str(strscore) + '(' + str(xmax - xmin) + ',' + str(ymax - ymin) + ')', (xmin,ymin), font, 1,(0,0,255),2)
                    #cv2.putText(re_img_np, strname + str(strscore) + '(' + str(xmax - xmin) + ',' + str(ymax - ymin) + ')', (xmin,ymin-10), font, 1,(0,255,0),2)
                    #cv2.putText(re_img_np, strname + str(strscore) + '(' + str(xmax - xmin) + ',' + str(ymax - ymin) + ')', (xmin,ymin-30), font, 1,(255,0,0),2)
                   
                   #cv2.putText(re_img_np, strname + str(strscore), (xmin,ymin+4), font, 1,(255,0,0),2)                    
                    #cv2.putText(re_img_np, str(strscore) + '('+str(xmin)+','+ str(ymin)+','+str(xmax)+','+str(ymax)+')', (xmin,ymin), font, 0.5,(0,0,255),1)
                print(os.path.join(output_image,file))
                cv2.imwrite(os.path.join(output_image,file), re_img_np)
    print(num_no_target)
    
if __name__ == '__main__':

    cfg_path = '/data/AI/zhangjing/detectron2/configs/Base-RetinaNet_zj.yaml'  
    weights_path = '' #'/data/AI/zhangjing/detectron2/output/oil/res101_1080_20191104_scale4_269/model_0059999.pth'
    input_imagespath = '/data/AI/zhangjing/detectron2/test_xml/img' #'/data/AI/zhangjing/train-data/14class/VOCdevkit2007/VOC2007/JPEGImages_benchmark_day269' 
    output_image     = '/data/AI/zhangjing/detectron2/test_xml/out'
    output_xmlpath   = '/data/AI/zhangjing/detectron2/test_xml/xml'
    main(cfg_path,weights_path,input_imagespath,output_image,output_xmlpath)