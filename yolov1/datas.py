#读取数据
import numpy as np
import os
import cv2
import xml.etree.ElementTree as ET
import copy

class load_data(object):
  def __init__(self,path,batch,CLASSES):
    #DevKit 是一个在 Windows 上帮助简化安装及使用 Ruby C/C++ 扩展如 RDiscount 和 RedCloth 的工具箱。
    self.devkit_path = path + '/VOCdevkit'
    self.data_path = self.devkit_path + '/VOC2007'
    self.img_size = 448
    self.batch = batch
    self.CLASS = CLASSES
    self.n_class = len(CLASSES)

    #将两个列表合并成一个字典 dict(zip())方法
    self.class_id = dict(zip(CLASSES,range(self.n_class)))#这句话给每个类进行编号
    self.id = 0
    self.run_this_data() #放在初始化中去调用其他的函数

    #存放当前训练的轮数
    self.epoch = 0

  def run_this_data(self):
    labels = self.load_label()
    np.random.shuffle(labels)
    self.truth_label = labels #把labels赋给类变量truth_labels
    return labels

  # 返回的是list列表：1，加载图片的地址2，其对应的groundtruth：7*7*25
  def load_label(self):
    # trianval.txt是用来训练和验证的图片文件的文件名列表
    path = self.data_path + '/ImageSets/Main/trainval.txt'

    #文件操作中的读写模式，'-r'是只读模式
    with open(path,'r') as f:
      #readline()读取一行内容，放到一个字符串变量，返回str类型。
      #readlines() 读取文件所有内容，按行为单位放到一个列表中，返回list类型。
      #Python strip() 方法用于移除字符串头尾指定的字符（默认为空格或换行符）或字符序列。
      index = [x.strip() for x in f.readlines()] #得到的index就是trainval.txt中的序号值
      #print('index: ',index)
    labels = []
    for i in index:
      la,num = self.load_xml(i) #这张图片中存在groundtruth：label：7*7*25；物体的数量
      if num == 0:
        continue
      img_name = self.data_path + '/JPEGImages/' + i + '.jpg' #加载图片地址
      labels.append({
          'img_name':img_name,
          'label':la
      })
    return labels

  # 返回：这张图片中存在groundtruth：label：7*7*25；物体的数量
  def load_xml(self,index):
    path = self.data_path + '/JPEGImages/' + index + '.jpg'
    xml_path = self.data_path + '/Annotations/' + index + '.xml'
    img = cv2.imread(path)
    #img.shape返回的是图像的行数（高），列数（宽）和channel
    h = 1.0 * self.img_size / img.shape[0]   # 448/h
    w = 1.0 * self.img_size / img.shape[1]   # 448/w

    label = np.zeros((7,7,25)) #label是一个7*7*25的数组
    tree = ET.parse(xml_path) #分析指定xml文件。得到tree
    objs = tree.findall('object') #找到第一标签为object的标签
    for i in objs:
      box = i.find('bndbox') #找到标签'bndbox'
      #这里的xy都是变成448*448下的了
      x1 = max(min((float(box.find('xmin').text) - 1) * w, self.img_size - 1),0) #注意这里的含义
      y1 = max(min((float(box.find('ymin').text) - 1) * h, self.img_size - 1),0)
      x2 = max(min((float(box.find('xmax').text) - 1) * w, self.img_size - 1),0)
      y2 = max(min((float(box.find('ymax').text) - 1) * h, self.img_size - 1),0)

      boxes = [(x1+x2)/2., (y1+y2)/2., x2-x1, y2-y1]      #box的中心点和宽，高,此处的宽和高没有归一化，依然是像素坐标
      cls_id = self.class_id[i.find('name').text.lower().strip()]

      x_id = int(boxes[0] * 7 / self.img_size)        #int()是向下取整，这里的x_id是从0开始的第几份
      y_id = int(boxes[1] * 7 / self.img_size)

      if label[y_id, x_id,0] == 1:
        continue
      label[y_id,x_id,0] = 1        # confidence 置信度，说明存在object
      label[y_id,x_id,1:5] = boxes;     # bounding box
      label[y_id,x_id,5 + cls_id] = 1   #物体分类,这个object是第cls_id类物体

    return label, len(objs)

  def load_image(self,PATH):
    im = cv2.imread(PATH)
    im = cv2.resize(im, (self.img_size,self.img_size))
    im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB).astype(np.float32)
    #im = np.multiply(1./255.,im) #图片是RGB形式，像素值再0~1之间
    im = (im / 255.0) * 2.0 - 1.0  #-1~1
    return im

  #最后主要就是用这个函数
  #返回：img：batch*448*448*3
  #   labels:batch*7*7*25
  def get_data(self):
    img = np.zeros((self.batch, self.img_size, self.img_size, 3)) #img是batch*448*448*3的数组
    labels = np.zeros((self.batch,7,7,25))              #labels是batch*7*7*25的数组
    times = 0
    #self.id = 0
    while times < self.batch:
      #truth_label是列表：labels.append({'img_name':img_name, 'label':la})
      img_name = self.truth_label[self.id]['img_name']        #把第self.id个'img_name'读出来
      img[times,:,:,:] = self.load_image(img_name)          #将图片的像素值存入img数组
      labels[times,:,:,:] = self.truth_label[self.id]['label']    #'label'是7*7*25的数组
      times += 1
      self.id += 1
      if self.id >= len(self.truth_label):
        np.random.shuffle(self.truth_label) #随机打乱truth_label
        self.id = 0
        self.epoch += 1
    
    return img, labels

if __name__ == '__main__':
  CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
              'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
              'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
              'train', 'tvmonitor']
  data_path = '.'
  test = load_data(data_path,45,CLASSES) #用法！
  print(CLASSES)
  a = range(20)
  print(a)
  print('class_id : \n',test.class_id)
  print('class_id[bird] : ', test.class_id['bird'])
  for i in range(5):
    img, labels = test.get_data()
    truth_labels = test.run_this_data()
    print('epoch = ',test.epoch)
    print('truth_label_image: ',truth_labels[1]['img_name'])
    print('truth_label:',truth_labels[1]['label'])
  print(img.shape,labels.shape)
  
