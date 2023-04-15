import random
import os, sys
from albumentations import *

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import warnings
warnings.filterwarnings('ignore')
import torch
import seaborn as sns
import torchvision
from torchvision import transforms
from torchvision.transforms import *
from scipy import stats
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import f1_score,recall_score
from torchvision.datasets import *

inform = {
    'name' : "hoyeon",
    'path' : "C:/Users/22668/Desktop/github/datasets/creatorcamp",
}

#경로에 가서 (이미지,레이블)의 튜플로 구성된 데이터를 가져옴
def get_ts_data(path):
  tr = transforms.Compose([
      transforms.ToTensor(), #각각의 픽셀값이 0,1사이인 텐서로 변환(min-max-norm 자동적용)
      transforms.Resize((224,224)), 
      #224,224고정
  ])
  dataset = torchvision.datasets.ImageFolder(path,transform = tr)
  return dataset


#데이터셋으로부터 인수로 받은 클래스의 이미지를 4-D tensor로 출력
def get_class_image(dataset,
              class_num, # 어떤 클래스?
              sampling = None, # 몇 개?
              seed_num=10):
  random.seed(seed_num)
  _tgts=pd.Series(dataset.targets)
  _class_idx=_tgts[_tgts == class_num].index.values.tolist()
  if sampling is not None:
    _get_idx = random.sample(_class_idx,sampling)
  else:
    _get_idx = _class_idx
  
  imgs_list = []
  for idx in _get_idx:
    imgs,_ = dataset[idx]
    imgs_list.append(imgs)
  imgs_list = torch.stack(imgs_list)
  return imgs_list  

#텐서로 변환된 하나의 이미지를 보여줌
def show_image(img):
  tmp_img=img.permute(1,2,0)
  plt.grid(False)
  plt.imshow(tmp_img.data)
  plt.axis('off')


#여러개의 이미지를 보여줌
def show_images(imgs #get_class_image의 출력이 매개변수로(즉,텐서들이 각각의 요소인 리스트)
                ):
  k = len(imgs)
  col=5
  row=int(k/5)
  if row == 0:
    row = 1
    col = k
    plt.subplots(row,col,figsize=(10,10))
  else:
    plt.subplots(row,col,figsize=(20,20))
  for idx,img in enumerate(imgs):
    plt.subplot(row,col,idx+1)
    show_image(img)
    plt.axis('off')

#여러개의 이미지를 보여줌
def show_images2(imgs #get_class_image의 출력이 매개변수로(즉,텐서들이 각각의 요소인 리스트)
                ):
  k = len(imgs)
  col=5
  row=int(k/5)
  if row == 0:
    row = 1
    col = k
    plt.subplots(row,col,figsize=(10,10))
  else:
    plt.subplots(row,col,figsize=(40,40))
  for idx,img in enumerate(imgs):
    plt.subplot(row,col,idx+1)
    show_image(img)
    plt.axis('off')

#이미지와 각 R,G,B 각각의 픽셀들의 분포를 보여주는 함수
def RGB_dist(image):
  plt.subplots(1,4,figsize=(40,10))
  plt.subplot(1,4,1)
  plt.title('images')
  show_image(image)

  plt.subplot(1,4,2)
  plt.title('R-Channel Distribution')
  sns.kdeplot(image[0].reshape(-1), color="red") #stat='density , 확률밀도함수 추정
  #plt.hist(image[0].reshape(-1),bins=100,color='red',alpha=0.8,stat='probability')
  #sns.kdeplot(image[0].reshape(-1),color="red")

  plt.subplot(1,4,3)
  plt.title('G-Channel Distribution')
  sns.kdeplot(image[1].reshape(-1), color="green") #stat='density , 확률밀도함수 추정



  plt.subplot(1,4,4)
  plt.title('B-Channel Distribution')
  sns.kdeplot(image[2].reshape(-1), color="blue") #stat='density , 확률밀도함수 추정


#kde : https://en.wikipedia.org/wiki/Kernel_density_estimation


def get_imgs(dataset,indices):
  imgs=[]
  for idx in indices:
    imgs.append(dataset[idx][0])
  return torch.stack(imgs)


def out_image(dataset, # 내보낼 데이터
              path, # 내보낼 경로
              indices # 내보낼 인덱스
              ):
  folder_stack = []
  for idx in indices:
    #1 이미지와 클래스 따로 저장
    img = dataset[idx][0]
    cls_num = str(dataset[idx][1])
    str_idx = str(idx)

    #2 저장할 폴더가 없으면 폴더 생성
    if cls_num not in folder_stack:
      folder_stack.append(cls_num)
      os.mkdir(path + '/' + cls_num)
    
    #3 저장할 경로와 파일이름 설정
    fold_name=str(folder_stack[-1])
    #os.path.join() 쓰면 작대기 자동추가해줌
    #다음에 이거 ㄱㄱ
    img_path = path + '/'  + fold_name + '/' + str_idx + '.jpg'

    #4 저장
    torchvision.utils.save_image(img,img_path)


#가정 : EDA 결과 입력이미지의 각 채널(R,G,B)의 확률밀도함수의 최댓값이 모두 threshold를 초과하면 추상이미지고. 아니면 실제사진(이상치)라고 가정
#threshold와 count가 hyperparameter!
def ck_outlier(img,
               threshold,          #확률밀도함수의 기준값
               sampling_num = 30): #Kde에서 sampling할 갯 수.

  #Kde추정하는 객체 만들기
  X = torch.linspace(0.5,1.1,sampling_num)
  #수정!
  #2022-10-08
  #검은색 배경을 가진 추상이미지는 별로 없고 실제사진이 검은색이므로 확률분포에서 0.5이상인 부분만 보면 됨!

  
  estr_0 = stats.gaussian_kde(img[0].reshape(-1), bw_method='silverman')
  estr_1 = stats.gaussian_kde(img[1].reshape(-1), bw_method='silverman')
  estr_2 = stats.gaussian_kde(img[2].reshape(-1), bw_method='silverman')

  #확률밀도함수의 y값 구해보기
  r_max = np.max(estr_0(X))
  g_max = np.max(estr_1(X))
  b_max = np.max(estr_2(X))
  #가정,R,G,B값 중 3개 모두 확률밀도의 최댓값이 5이상이면 추상이미지(픽토그램,일러스트레이션...)라고 가정
  count=0
  if r_max>threshold:
    count+=1
  if g_max>threshold:
    count+=1
  if b_max>threshold:
    count+=1
  if count >=3:
    return False #추상이미지는 아웃라이어 아니므로 False
  else:
    return True 
  
def predict_outlier(imgs,threshold): #다수의 이미지들이 이상치인지 아닌지 예측해주는 함수!
  if type(threshold) is int or type(threshold) is float or type(threshold) is np.float64:
    pred_y=[]
    for img_idx in range(imgs.shape[0]):
      _pdtn = ck_outlier(imgs[img_idx],threshold)
      if _pdtn == False: #추상이미지라면 0 (이상치가 아니므로 0)
        pred_y.append(0)
      else: #실제사진이라면 1(이상치이므로1)
        pred_y.append(1)
    pred_y=np.array(pred_y).reshape(-1,1)
    return pred_y
  else:
    pred_y_list=[]
    for thold in threshold:
      pred_y=[]
      for img_idx in range(imgs.shape[0]):
        _pdtn = ck_outlier(imgs[img_idx],thold)
        if _pdtn == False: #추상이미지라면 0 (이상치가 아니므로 0)
          pred_y.append(0)
        else: #실제사진이라면 1(이상치이므로1)
          pred_y.append(1)
      pred_y=np.array(pred_y).reshape(-1,1)      
      pred_y_list.append(np.array(pred_y))
    return pred_y_list


#정확도체크,레이블이 없으므로 3직접 만들어주는 과정이 필요해요.
def ac_ck(imgs,labels,miss_idx=False):
  ck=[]
  tmp=0
  #print('start')
  for img_idx in range(len(imgs)):
    print(tmp)
    if ck_outlier(imgs[img_idx]) == True:
      ck.append(True)
    else:
      ck.append(False)
    tmp+=1
  k = imgs.shape[0]
  ck = np.array(ck)
  labels = np.array(labels)
  tmp=(ck==labels)
  accuracy=sum(tmp)/len(tmp)
  if miss_idx == False:
    return accuracy
  else:
    miss_list=[]
    for idx in range(len(tmp)):
      if tmp[idx] == False:
        miss_list.append(idx)
    return accuracy,miss_list