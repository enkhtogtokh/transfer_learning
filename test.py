import time

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
 
import numpy as np
import mxnet as mx
from mxnet import autograd, gluon
import gluoncv as gcv
from gluoncv.utils import download, viz

#video
from gluoncv import utils
from decord import VideoReader
import cv2

 

ctx = mx.cpu()

def get_dataloader(net, train_dataset, data_shape, batch_size, num_workers):
    from gluoncv.data.batchify import Tuple, Stack, Pad
    from gluoncv.data.transforms.presets.ssd import SSDDefaultTrainTransform
    width, height = data_shape, data_shape
    # use fake data to generate fixed anchors for target generation
    with autograd.train_mode():
        _, _, anchors = net(mx.nd.zeros((1, 3, height, width)))
    batchify_fn = Tuple(Stack(), Stack(), Stack())  # stack image, cls_targets, box_targets
    train_loader = gluon.data.DataLoader(
        train_dataset.transform(SSDDefaultTrainTransform(width, height, anchors)),
        batch_size, True, batchify_fn=batchify_fn, last_batch='rollover', num_workers=num_workers)
    return train_loader
 

def test():

    #Test
    classes = ['mercedes']
    test_url = 'https://blog.mercedes-benz-passion.com/wp-cb4ef-content/uploads/EQC_SPOT_Mercedes_Antoni_Garage_2019_Spot_EQC_Campaign_EN_DE_11.jpg'
    download(test_url, 'benz_test.jpg')
    classes = ['mercedes','person','car']
    net = gcv.model_zoo.get_model('ssd_512_mobilenet1.0_custom', classes=classes, pretrained_base=False, ctx=ctx)
    net.load_parameters('ssd_512_mobilenet1.0_benz100.params')
    net.hybridize()
    
    x, image = gcv.data.transforms.presets.ssd.load_test('benz_test.jpg', 512)
    cid, score, bbox = net(x)
    ax = viz.plot_bbox(image, bbox[0], score[0], cid[0], class_names=classes)
    plt.show()

def test_video():
    

    fourcc = cv2.VideoWriter_fourcc(*"MJPG") 
	
    video_writer = None

    classes = ['mercedes','person','car']
    net = gcv.model_zoo.get_model('ssd_512_mobilenet1.0_custom', classes=classes, pretrained_base=False, ctx=ctx)
    net.load_parameters('ssd_512_mobilenet1.0_benz100.params')
    net.hybridize()
    #Test video
    #url ='https://www.youtube.com/watch?v=7iGFH5HbWGM' #'https://github.com/bryanyzhu/tiny-ucf101/raw/master/abseiling_k400.mp4'
    #video_fname = utils.download(url)
    vr = VideoReader("test3.mov")
    duration = len(vr)
    print('The video contains %d frames' % duration)
     
    root_path = "video_rec_tmp/" 
    cap = cv2.VideoCapture("test3.mov")
    cnt = 0
    while(cap.isOpened()):
         ret, frame = cap.read()
  
          
         imp_path = root_path+str(cnt)+".jpg"
         cv2.imwrite(imp_path,frame)    
         cnt += 1 
         x, image = gcv.data.transforms.presets.ssd.load_test(imp_path, 512)
         cid, score, bbox = net(x)

         if(cnt==500000000):
           ax = viz.plot_bbox(image, bbox[0], score[0], cid[0], class_names=classes)
           plt.show()
       
         
         

         
         if not ret: break

         #ax = viz.plot_bbox(image, bbox[0], score[0], cid[0], class_names=classes) score[0]
         viz.cv_plot_bbox(image, bbox[0], score[0], cid[0], class_names=classes)
         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

         if video_writer is None:
            height , width , layers =  image.shape 
            video_writer = cv2.VideoWriter("video_rec_tmp/mercedes_rec15min.avi",fourcc, 30, (width,height))
            print(video_writer)

         cv2.imshow('Mercedes Recognition & Track CV',image)
         print(cnt)
         if cv2.waitKey(1) & 0xFF == ord('q'):
           break
         cv2.imwrite(imp_path,image)  
         video_writer.write(image) 
         


    cap.release()
    video_writer.release()

    

    
 
   
    

def test_image_bb(file_path):
    img = mx.image.imread(file_path)
    print(img)
    classes = ['mercedes']  # only one foreground class here
    bb =  [615, 1077,840, 1298]
    all_boxes = np.array([bb, bb])
    print('label:', bb)
    labels = ["","","",""]
    # display image and label
    ax = viz.plot_bbox(img, all_boxes, labels=None, class_names=classes)
    plt.show()

from os import walk
import os

def record_video():
    video_writer = None
    for i in range (0,2000):
         
            imp_path = "video_rec/"+str(i)+".jpg"
            if not os.path.exists(imp_path):
                continue
             
            print(imp_path)
            x, image = gcv.data.transforms.presets.ssd.load_test(imp_path, 512)

    
            if video_writer is None:
               fourcc = cv2.VideoWriter_fourcc(*"MJPG")
               video_writer = cv2.VideoWriter("training/mercedes_rec.avi", fourcc, 35,
			    (image.shape[1], image.shape[0]), True)

           
               height , width , layers =  image.shape 
            
               print(video_writer,width,height)
         
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            video_writer.write(image) 
    video_writer.release()

    
 
test_video()
 