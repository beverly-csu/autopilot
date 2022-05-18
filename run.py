from Detector import *

modelURL = "http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet152_v1_1024x1024_coco17_tpu-8.tar.gz"
# modelURL = "http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz"
classFile = "coco.names"
imagePath = "C:/Users/Danya/Desktop/brod.jpg"
videoPath = "C:/Users/Danya/Downloads/test.mp4"
# videoPath = 0
threshold = 0.5
detector = Detector()
detector.readClasses(classFile)
detector.downloadModel(modelURL)
detector.loadModel()
# detector.predictImage(imagePath, threshold)
detector.predictVideo(videoPath, threshold)