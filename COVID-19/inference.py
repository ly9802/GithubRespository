from __future__ import print_function
import torch
import torch.nn as nn
from torchvision import models
import torch, os, copy, time, pickle
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image
import pandas as pd
from torchvision.datasets.folder import IMG_EXTENSIONS
from torchvision.datasets import ImageFolder
from sklearn.metrics import confusion_matrix
from sklearn.metrics import average_precision_score,precision_recall_curve,accuracy_score;
from sklearn.metrics import precision_score,accuracy_score,recall_score,roc_auc_score;
import glob, pickle
import seaborn as sn
import argparse
from utils.rela import correlateCAM
parser = argparse.ArgumentParser(description='lyCOVID-19 Detection from X-ray Images')
parser.add_argument('--test_covid_path', type=str, default='../../../dataset/DeepCovidDataset/val/covid/',
                      help='COVID-19 test samples directory')
parser.add_argument('--test_non_covid_path', type=str, default='../../../dataset/DeepCovidDataset/val/non/',
                      help='Non-COVID test samples directory')
parser.add_argument('--trained_model_path', type=str, default='./net_model/covid_resnet50_epoch100_balance.pt',
                      help='The path and name of trained model')

parser.add_argument('--cut_off_threshold', type=float, default= 0.205,
                    help='cut-off threshold. Any sample with probability higher than this is considered COVID-19 (default: 0.2)')
parser.add_argument('--batch_size', type=int, default=20,
                    help='input batch size for training (default: 20)')
parser.add_argument('--num_workers', type=int, default=0,
                    help='number of workers to train (default: 0)')
args = parser.parse_args()
def find_sens_spec( covid_prob, noncovid_prob, thresh):
    '''
    numpy array covid_prob
    numpy array noncovid_prob
    float thresh
    '''
    sensitivity= (covid_prob >= thresh).sum()   / (len(covid_prob)+1e-10);
    #这里,thresh=默认0.2, 那么在预测的时候,不一定在0的位置是最大比例, 比如(0.4,0.6) ,很显然,预测是label1, 因为最大值0.6在位置1上, 但是
    #在这里统计的时候,把它算上预测label 是0的.
    specificity= (noncovid_prob < thresh).sum() / (len(noncovid_prob)+1e-10);
    #同理, 这里thresh=0.2, 假如是(0.4,0.6)的情况, 这里认为是false, 不统计进入,
    #按照它这个算法, 不是说哪个位置上大算哪个label,而是第一个位置上0位置上的比例超过thresh 0.2 就算预测 label 为0
    print("sensitivity= %.3f, specificity= %.3f" %(sensitivity,specificity))
    return sensitivity, specificity
def image_loader(image_name):
    """load image, returns cuda tensor
    str image_name
    return a tensor
    """
    image = Image.open(image_name).convert("RGB")
    image = loader(image).float()
    image = Variable(image, requires_grad=True)
    image = image.unsqueeze(0)  #this is for VGG, may not be needed for ResNet, 在0维加了一个维度
    # tensor image
    return image
def image_to_PIL(image_name):
    image = Image.open(image_name).convert("RGB");
    return trans(image);
imsize= 224
loader = transforms.Compose([transforms.Resize((imsize,imsize)),
                                 transforms.CenterCrop(imsize),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                 ])
trans=transforms.Compose([
        transforms.Resize((imsize,imsize)),
        transforms.CenterCrop(imsize),
    ])
if __name__=="__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = ['covid','non']

    net_path="./net_model/net_resnet18_epoch100.pt";
    fc_path="./net_model/fc_resnet18_epoch100.pt";
    net=torch.load(net_path,map_location='cpu');
    fc=torch.load(fc_path,map_location='cpu');
    net.cpu().eval();
    fc.cpu().eval();

    sm = torch.nn.Softmax(dim=1)
    test_covid  = glob.glob("%s*" %args.test_covid_path)
    test_non    = glob.glob("%s*" %args.test_non_covid_path);
    l=0;
    for i in range(len(test_non)):
        for j in os.listdir(test_non[i]):
            l=l+1;

    covid_pred= np.zeros([len(test_covid),1]).astype(int); #non_pred  = np.zeros([len(test_non),1]).astype(int) # 全是int
    non_pred  = np.zeros([l,1]).astype(int)
    covid_prob= np.zeros([len(test_covid),1]) #non_prob   = np.zeros([len(test_non),1])# 全是float,全是0, 形状定好.比如test_covid, 100,1
    non_prob   = np.zeros([l,1])
    covid_num=0;covid_num2=0;
    
    for i in range(len(test_covid)):
        # len(test_covid)=100
        cur_img = image_loader(test_covid[i]).cpu();
        img_croped_obj = image_to_PIL(test_covid[i]);
        # tensor cur_img;str test_covid[i]
        img_featuremap= net(cur_img);# shape(1,512,7,7)
        batchsize, channels, h,w=img_featuremap.shape;
        model_output=fc(img_featuremap);# model_output.shape (1,2)
        cur_pred = model_output.max(1, keepdim=True)[1].cpu()  # 取位置 返回是个tuple, index=0 是指最大值是多少, index=1指最大值的位置
        # cur_pred 虽然是tensor, 实际上是int, 就是位置# 这里按最大值来给label
        cur_prob = sm(model_output).cpu()  # sm是softmax的一个实例,softmax是一个类
        # softmax 就是计算model_output的每个位置的占比, 返回也是一个tensor,shape是(1,2)
        covid_prob[i, :] = cur_prob.data.numpy()[0, 0];  # [0,0]是指第一行第一个数, [0,1]是指第一行第二个数, 总共就2个数.

        linear_weight=fc.fc.weight.data# weight 是parameter对象,但是data是tensor对象, shape(2,512)
        indice=fc.temp_one.cpu();

        if float(cur_prob.data.numpy()[0, 0]) > args.cut_off_threshold:# show this instance belongs to COVID19
            covid_num2 = covid_num2 + 1;
            img_croped_obj.save("./output0/croped_img.jpg");
            img_croped_obj.save("./output/covid19/croped_img" + str(i) + ".jpg");
            temp=img_featuremap.contiguous().view(batchsize,channels,-1).cpu();#( batchsize, channels,h*w);
            new_featuremap=torch.mul(temp,indice);# shape(1,512,49)
            new_featuremap=new_featuremap.contiguous().view(batchsize,channels,h,w);
            CAMs = correlateCAM.calc_cam(new_featuremap.detach().numpy(), linear_weight.detach().numpy(), 0);
            heatmap = cv2.applyColorMap(CAMs, cv2.COLORMAP_JET);
            img_cv2_obj = cv2.imread("./output0/croped_img.jpg");
            result = heatmap * 0.3 + img_cv2_obj * 0.6;
            #cv2.imwrite("./output/covid19/heatmap" + str(i) + ".jpg", result);
            #cv2.imwrite("./output/heatmap/heatmap" + str(i) + ".jpg", heatmap);
        else: # this instance does not belong to COVID19,but it indeed is , false negative
            img_croped_obj.save("./output0/croped_img.jpg");
            img_croped_obj.save("./output/falseNegative/croped_img" + str(i) + ".jpg")
            temp = img_featuremap.contiguous().view(batchsize, channels, -1).cpu();
            new_featuremap = torch.mul(temp, indice);
            new_featuremap = new_featuremap.contiguous().view(batchsize, channels, h, w);

            CAMs = correlateCAM.calc_cam(new_featuremap.detach().numpy(), linear_weight.detach().numpy(), 1);  # idx[0] is the maximum
            heatmap = cv2.applyColorMap(CAMs, cv2.COLORMAP_JET);
            img_cv2_obj = cv2.imread("./output0/croped_img.jpg");
            result = heatmap * 0.3 + img_cv2_obj * 0.6;
            #cv2.imwrite("./output/fasleNegative/heatmap" + str(i) + ".jpg", result);

        print("%03d Covid predicted label:%s" % (i + 1, class_names[int(cur_pred.data.numpy())]))
    print("total num of covid (use thresh)", covid_num2);
    k = 0;
    for i in range(len(test_non)):
        for j in os.listdir(test_non[i]):
            cur_img= image_loader(os.path.join(test_non[i],j)).cpu()
            img_fmap=net(cur_img);
            model_output= fc(img_fmap)
            cur_pred = model_output.max(1, keepdim=True)[1].cpu()# 选位置
            cur_prob = sm(model_output).cpu()
            non_prob[k,:]= cur_prob.data.numpy()[0,0];# [0,0]是指第一行第一个数, [0,1]是指第一行第二个数, 总共就2个数. # 这里就指类别是0的百分比
            k=k+1;# 还是[0,0]选小的那个数
            print("%03d Non-Covid predicted label:%s" %(k, class_names[int(cur_pred.data.numpy())]) )
    thresh= args.cut_off_threshold
    sensitivity_40, specificity= find_sens_spec( covid_prob, non_prob, thresh)
    covid_pred = np.where( covid_prob  >thresh, 1, 0)# 满足条件 就是1, 否则就是0
    non_pred   = np.where( non_prob    >thresh, 1, 0)# 满足条件 就是1 ,否则就是0
    covid_list  = [int(covid_pred[i]) for i in range(len(covid_pred))]# 计算总数 应该是100
    #print("how many covid 19",covid_list.count(1));
    covid_count = [(x, covid_list.count(x)) for x in set(covid_list)]
    non_list= [int(non_pred[i]) for i in range(len(non_pred))]
    non_count = [(x, non_list.count(x)) for x in set(non_list)]
    y_pred_list= covid_list+non_list
    y_test_list= [1 for i in range(len(covid_list))]+[0 for i in range(len(non_list))]
    y_pred= np.asarray(y_pred_list, dtype=np.int64)
    y_test= np.asarray(y_test_list, dtype=np.int64);
    val_acc1=accuracy_score(y_test,y_pred);
    precision=precision_score(y_test,y_pred,pos_label=1);
    roc_auc=roc_auc_score(y_test, y_pred);
    recall=recall_score(y_test, y_pred,pos_label=1);
    average_precision=average_precision_score(y_test,y_pred);
    true_pos, false_neg, false_pos, true_neg = confusion_matrix(y_test, y_pred).ravel();
    print("The precision:{}, recall:{}, roc_auc:{}, val_acc1:{}".format(precision,recall,roc_auc,val_acc1),flush=True);
    print("The average precision is {}".format(average_precision));
    print("The sensitivity: {}, specificity: {}\n".format(recall,specificity));
    print("The Confusion Matrix as follows:\n");
    print("               predict_Covid19, predict_nonCovid\n");
    print("true_Covid19 :   {}                {}\n".format(true_neg,false_pos));
    print("true_nonCovid19: {}                {}\n".format(false_neg,true_pos));
    
    cnf_matrix = confusion_matrix(y_test, y_pred)
    np.set_printoptions(precision=2)
    df_cm = pd.DataFrame(cnf_matrix, index = [i for i in class_names],
                                     columns = [i for i in class_names])
    ax = sn.heatmap(df_cm, cmap=plt.cm.Blues, annot=True, cbar=False, fmt='g', xticklabels= ['Non-COVID','COVID-2019'], yticklabels= ['Non-COVID','COVID-2019'])
    ax.set_title("Confusion matrix")
    plt.savefig('./confusion_matrix.png') #dpi = 200


