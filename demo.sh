'''python demo/demo.py  \
--config-file configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml \
--input input.jpg \
--opts MODEL.WEIGHTS detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl \
--output input_res.jpg


python demo/demo.py  \
--config-file configs/COCO-Detection/retinanet_R_101_FPN_3x.yaml \
--input input.jpg \
--output input_res.jpg \
--opts MODEL.WEIGHTS model_final_RetinaNet_R101.pkl '''

python demo/demo.py  \
--config-file configs/Base-RetinaNet_zj.yaml \
--input /data/AI/zhangjing/detectron2/test_xml/img/20180921_50E54921F7C4_00008.jpg \
--output input_res.jpg \
--opts MODEL.WEIGHTS /data/AI/zhangjing/detectron2/output/oil/res101_1080_20191104_scale4_20191125/model_0059999.pth
#output/oil/test/model_0059999.pth
