from Detector import *

# modelURL = "http://download.tensorflow.org/models/object_detection/tf2/20200711/mask_rcnn_inception_resnet_v2_1024x1024_coco17_gpu-8.tar.gz"
modelURL = "http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz"
# imagePath = r"road.jpeg"
classFiles = "coco.names"
videopath = 0
detector = Detector()
threshold = 0.5
detector.readclasses(classFiles)
detector.downloadModel(modelURL)
detector.loadModel()
# detector.predictImage(imagePath,threshold)
detector.predictvideos(videopath,threshold)
