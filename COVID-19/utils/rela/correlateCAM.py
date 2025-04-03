import numpy as np
import cv2
import torch
def calc_cam(features,weight_pooling, class_idx):
    # numpy array: features ( feature map, shape( batchsize, channels, height, width)
    # numpy array :weight_pooling is a matrix shape ( num_classes, channels)
    # int class_idx
    size_upsample = (224, 224);

    batchsize,channels_num, h, w = features.shape;
    #features.shape:(1,2048,7,7)
    #weight_pooling: (2,2048)
    feature_matrix=features.reshape((channels_num,h*w));
    #weight_matrix=weight_pooling[class_idx].contiguous().view(1,-1);
    #cam=torch.mm(weight_matrix,feature_matrix)
    cam=weight_pooling[class_idx].dot(feature_matrix);
    cam=cam.reshape(h,w);# cam.shape(7,7)
    cam=cam-np.min(cam);
    cam_img=cam/np.max(cam);
    cam_img1=np.uint8(255*cam_img); #ndarray cam_img: 7*7
    print("The original cam is\n", cam_img1);
    cams=cv2.resize(cam_img1,size_upsample,interpolation=cv2.INTER_CUBIC);
    #print("The expanded image is \n",cams);
    return cams
    
def filter(twoD_array, mean):
    for i in range(twoD_array.shape[0]):
        for j in range(twoD_array.shape[1]):
            if twoD_array[i,j]<mean:
                twoD_array[i,j]=0;
    return twoD_array;

def mask_to_cam(mask):
    # ndarray shape must be (1,7,7);
    size_upsample=(224, 224);
    _,h,w=mask.shape;
    mask=mask.reshape(h,w);
    mask=mask-np.min(mask);
    cam_img_mask=mask/np.max(mask);
    cam_img1_mask=np.uint8(255*cam_img_mask);
    #print("the Mask to cam is \n",cam_img1_mask);
    cams=cv2.resize(cam_img1_mask,size_upsample,interpolation=cv2.INTER_CUBIC);
    return cams