import numpy as np
import tensorflow as tf
import cv2
from google.colab.patches import cv2_imshow
#slim = tf.contrib.slim
#tf.reset_default_graph() #重置网络
'''
    7 x 7网格，如果某个物体目标的中心落在某个网格之中，那么这个网格就负责检测这个物体
    每个网格会有两个box， 这个网格的confidence置信度： confidence[0,1]，物体中心落在这个网格，confidence = 1，否则confidence = 0
    class scores: score = confidence * IOU: 如果没有物体落在这个box中 那么sore = 0
    全连接层fc = [batch,7,7,30]   2boxes*(x,y,w,h,c)--->2*5=10, 20--->类别的置信度
    ---> 7*7*2 = 98 boxes, 
'''
class Yolo(object):

  def __init__(self,is_training=True):
    self.classes = ["aeroplane", "bicycle", "bird", "boat", "bottle",
              "bus", "car", "cat", "chair", "cow", "diningtable",
              "dog", "horse", "motorbike", "person", "pottedplant",
              "sheep", "sofa", "train","tvmonitor"]
    self.C = len(self.classes) #物体类别数量
    #offset 偏移量
    self.x_offset = np.transpose(np.reshape(np.array([np.arange(7)] * 7 * 2), [2,7,7]), [1,2,0]) # (7, 7, 2)
    # x_offset =
    '''array(7个[[[0, 0],
          [1, 1],
          [2, 2],
          [3, 3],
          [4, 4],
          [5, 5],
          [6, 6]],'''
    self.y_offset = np.transpose(self.x_offset, [1,0,2]) #0 1 2 3 4 5 6
    #x,y shape = (7,7,2)
    self.threshold = 0.2 #confidence score #格子有目标的置信度阈值
    self.iou_threshold = 0.5
    self.max_output_size = 10
    self.img_shape = (448,448)

    self.batch_size = 45 

    self.coord_scale = 5.
    self.noobject_scale = 1.
    self.object_scale = 1.
    self.class_scale = 2.

    self.images = tf.placeholder(tf.float32, [None, 448, 448, 3])

    if is_training: # 是否训练
      
      self.labels = tf.placeholder(tf.float32,[None, 7, 7, 25]) 
      self.logits = self._build_net()      
      #self.total_loss = self.loss_layer(self.logits,self.labels) 
      self.loss_layer(self.logits,self.labels)
      self.total_loss = tf.losses.get_total_loss()
      print('Training!!!')
      #用来显示标量信息
      tf.summary.scalar('total_loss', self.total_loss)

############  网络部分   ###############
  #注意！！！weight的初始化非常的重要，如果stddev设置的比较大就会出现Nan

  def leak_relu(self, x, alpha = 0.1):
    return tf.maximum(alpha * x, x)
  
  # net = self._conv_layer(x, 64, 7, 2, 'conv_2')
  def _conv_layer(self, x, num_filters, filter_size, stride, padding='SAME', scope=None):
    #print('\n',scope,padding)
    in_channels = x.get_shape().as_list()[-1] #x是tensor，x.get_shape()也返回一个tensor。用as_list()转换为list列表，[-1]取列表的最后一位
    weight = tf.Variable(tf.truncated_normal(
        [filter_size, filter_size, in_channels, num_filters],
        stddev = 0.01), name = 'weights')
    
    #tf.zeros([num_filters,])和tf.zeros([num_filters])似乎没什么区别，不知道逗号有啥用
    bias = tf.Variable(tf.zeros([num_filters,]), name = 'biases')
    #b_conv1 = tf.Variable(tf.constant(0.1,shape=[num_filters]), name='biases') #num_filters bias for num_filters outputs

    #pad_size = filter_size // 2 #向下取整
    #pad_mat = np.array([[0,0], [pad_size, pad_size], [pad_size, pad_size], [0,0]])
    #x_pad = tf.pad(x, pad_mat) #给tensor进行填充
    #conv = tf.nn.conv2d(x_pad, weight, strides=[1, stride, stride, 1], padding='VALID', name=scope)
    conv = tf.nn.conv2d(x, weight, strides=[1, stride, stride, 1], padding=padding, name=scope) + bias
    #output = self.leak_relu(tf.nn.bias_add(conv, bias))
    output = self.leak_relu(conv)

    return output

  def _fc_layer(self, x, num_out, activation=None, scope=None):
    
    num_in = x.get_shape().as_list()[-1]
    weight = tf.Variable(tf.truncated_normal([num_in, num_out],stddev = 0.01), name = 'weights')
    bias = tf.Variable(tf.zeros([num_out,]), name='biases')
    #bias = tf.Variable(tf.constant(0.1,shape=[num_out]), name='biases')#换成这个写法就会崩，不知原因
    output = tf.nn.xw_plus_b(x, weight, bias, name=scope)
    
    #如果需要激活函数，使用leak_relu
    if activation: 
      output = activation(output)

    return output
  
  def _maxpool_layer(self, x, pool_size, stride):
    output = tf.nn.max_pool(x, [1,pool_size, pool_size, 1],
                 strides=[1, stride, stride, 1], padding='SAME')
    return output

  def _flatten(self, x):
    '''flatten the x'''
    tran_x = tf.transpose(x, [0,3,1,2], name='trans_31') # channel first mode 变为NCHW模式
    nums = np.product(x.get_shape().as_list()[1:])# x的shape变为list，[1:]是指从第二位到最后一位，np.product是将第二位到最后一位的数相乘
    return tf.reshape(tran_x, [-1, nums]) #-1是不管第一通道
  
  #网络可能出问题了
  def _build_net(self):  #24个卷积，3个全连接
    x = self.images
    
    with tf.variable_scope('yolo'):
      
      #第一层
      net = tf.pad(
            x, np.array([[0, 0], [3, 3], [3, 3], [0, 0]]),
            name='pad_1') 
      #def _conv_layer(self, x, num_filters, filter_size, stride, padding='SAME', scope=None)     
      
      with tf.variable_scope('conv_2'):
        net = self._conv_layer(net, 64, 7, 2, padding='VALID',scope='conv_2')  #因为步长为2，所以输出的图像尺寸为输入的1/2
      
      net = self._maxpool_layer(net, 2, 2) #第3层maxpooling
      
      with tf.variable_scope('conv_4'):
        net = self._conv_layer(net, 192, 3, 1, scope='conv_4')
      net = self._maxpool_layer(net, 2, 2) #第5层maxpooling      
      with tf.variable_scope('conv_6'):
        net = self._conv_layer(net, 128, 1, 1, scope='conv_6')
      with tf.variable_scope('conv_7'):
        net = self._conv_layer(net, 256, 3, 1, scope='conv_7')
      with tf.variable_scope('conv_8'):
        net = self._conv_layer(net, 256, 1, 1, scope='conv_8')
      with tf.variable_scope('conv_9'):
        net = self._conv_layer(net, 512, 3, 1, scope='conv_9')
      net = self._maxpool_layer(net, 2, 2) #第10层maxpooling
      
      #从这开始变坏了~~~这里还可以接受
      with tf.variable_scope('conv_11'):
        net = self._conv_layer(net, 256, 1, 1, scope='conv_11')
      
      #可以接受
      with tf.variable_scope('conv_12'):
        net = self._conv_layer(net, 512, 3, 1, scope='conv_12')      
      #这层开始变得很差 NAN了
      with tf.variable_scope('conv_13'):
        net = self._conv_layer(net, 256, 1, 1, scope='conv_13')
      
      with tf.variable_scope('conv_14'):
        net = self._conv_layer(net, 512, 3, 1, scope='conv_14')
      
      with tf.variable_scope('conv_15'):
        net = self._conv_layer(net, 256, 1, 1, scope='conv_15')
      
      with tf.variable_scope('conv_16'):
        net = self._conv_layer(net, 512, 3, 1, scope='conv_16')
      with tf.variable_scope('conv_17'):
        net = self._conv_layer(net, 256, 1, 1, scope='conv_17')
      with tf.variable_scope('conv_18'):
        net = self._conv_layer(net, 512, 3, 1, scope='conv_18')
      with tf.variable_scope('conv_19'):
        net = self._conv_layer(net, 512, 1, 1, scope='conv_19')
      with tf.variable_scope('conv_20'):
        net = self._conv_layer(net, 1024, 3, 1, scope='conv_20')
      net = self._maxpool_layer(net ,2 ,2) #第21层maxpooling
      with tf.variable_scope('conv_22'):
        net = self._conv_layer(net, 512, 1, 1, scope='conv_22')
      with tf.variable_scope('conv_23'):
        net = self._conv_layer(net, 1024, 3, 1, scope='conv_23')
      with tf.variable_scope('conv_24'):
        net = self._conv_layer(net, 512, 1, 1, scope='conv_24')
      with tf.variable_scope('conv_25'):
        net = self._conv_layer(net, 1024, 3, 1, scope='conv_25')
      with tf.variable_scope('conv_26'):
        net = self._conv_layer(net, 1024, 3, 1, scope='conv_26')
      #27
      net = tf.pad(
              net, np.array([[0, 0], [1, 1], [1, 1], [0, 0]]),
              name='pad_27')     
      #padding, 第一个维度batch和第四个维度channels不用管，只padding卷积核的高度和宽度

      with tf.variable_scope('conv_28'):
        net = self._conv_layer(net, 1024, 3, 2, padding='VALID', scope='conv_28')
      with tf.variable_scope('conv_29'):
        net = self._conv_layer(net, 1024, 3, 1, scope='conv_29')
      with tf.variable_scope('conv_30'):
        net = self._conv_layer(net, 1024, 3, 1, scope='conv_30')
      
      #31 这一层已经包含在32里了     
      #转置，由[batch, image_height,image_width,channels]变成[bacth, channels, image_height,image_width]
      #net = tf.transpose(net, [0, 3, 1, 2], name='trans_31') 
      
      #32 这层是不是有问题
      net = self._flatten(net) # NCHW 并变为向量
      
      
      # fc_layer似乎和weight的初始化有关系，slim的效果就好点
      with tf.variable_scope('fc_33'):
        net = self._fc_layer(net, 512, activation=self.leak_relu,scope='fc_33')
      with tf.variable_scope('fc_34'):
        net = self._fc_layer(net, 4096, activation=self.leak_relu,scope='fc_34')
      
      #35 drop out 
      #防止过拟合，添加dropout层
      #keep_prob = tf.placeholder(tf.float32)
      net = tf.nn.dropout(net,0.5,name='dropout_35') #可以用~~~
      #layer_drop
      #net = slim.dropout(  #dropout，防止过拟合
      #      net, keep_prob=0.5, is_training=True,
      #      scope='dropout_35')
      
      with tf.variable_scope('fc_36'):
        net = self._fc_layer(net,7*7*30, scope='fc_36')

      return net

############ IOU和损失函数 ##########

  def calc_iou(self, boxes1, boxes2, scope='iou'): #要注意这两个box的4个参数的定义是什么
    '''calculate ious
    这个函数的主要作用是计算两个 bounding box 之间的 IoU。输入是两个 5 维的bounding box,输出的两个 bounding Box 的IoU     
    Args:
      bboxes1: 5-D tensor [BATCH_SIZE, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL, 4] ====> (x_center, y_center, w, h)
      bboxes2: 5-D tensor [BATCH_SIZE, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL, 4] ====> (x_center, y_center, w, h)
      注意这里的参数x_center, y_center, w, h都是归一到[0,1]之间的，分别表示预测边界框的中心相对整张图片的坐标，宽和高
    Return:
      iou: 4-D tensor [BATCH_SIZE, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
    '''
    #bboxes1 = np.transpose(bboxes1)  #实在看不懂这个transpose是在干啥，这里的代码是ssd的
    #bboxes2 = np.transpose(bboxes2)
    
    with tf.variable_scope(scope):
      # transform (x_center, y_center, w, h) to (x1, y1, x2, y2)
      bboxes1 = tf.stack([ boxes1[..., 0] - boxes1[..., 2] / 2.0,  #左上角x
                  boxes1[..., 1] - boxes1[..., 3] / 2.0,  #左上角y
                  boxes1[..., 0] + boxes1[..., 2] / 2.0,  #右下角x
                  boxes1[..., 1] + boxes1[..., 3] / 2.0],  #右下角y
                  axis=-1)

      bboxes2 = tf.stack([ boxes2[..., 0] - boxes2[..., 2] / 2.0,
                  boxes2[..., 1] - boxes2[..., 3] / 2.0,
                  boxes2[..., 0] + boxes2[..., 2] / 2.0,
                  boxes2[..., 1] + boxes2[..., 3] / 2.0],
                  axis=-1)
      
      # calculate the left up point & right down point
      #lu和rd就是分别求两个框相交的矩形的左上角的坐标和右下角的坐标，因为对于左上角，
      #选择的是x和y较大的，而右下角是选择较小的，可以想想两个矩形框相交是不是这中情况
      lu = tf.maximum(bboxes1[..., :2], bboxes2[..., :2])       #两个框相交的矩形的左上角(x1,y1)
      rd = tf.minimum(bboxes1[..., 2:], bboxes2[..., 2:])       #两个框相交的矩形的右下角(x2,y2)
      
      #intersection 就是求相交矩形的长和宽，所以有rd-lu，相当于x1-x2和y1-y2
      #使用tf.maximum是因为要删除那些不合理的框，比如两个框没有交集
      #就会出现左上角比右下角坐标值还大的情况
      intersection = tf.maximum(0.0, rd - lu)
      #inter_square是求交集矩形面积
      inter_square = intersection[...,0] * intersection[...,1]

      #计算两个框的面积，即用两个框的长和宽相乘
      square1 = boxes1[...,2] * boxes1[...,3]
      square2 = boxes2[...,2] * boxes2[...,3]

      #union_square就是并集面积
      union_square = tf.maximum(square1+square2-inter_square, 1e-10)

      iou = tf.clip_by_value(inter_square / union_square, 0.0, 1.0)

    print('iou', iou)
    return iou

    #print('iou计算！ 0')
    #print('boxes1:',bboxes1)
    #计算两个box的交集：交集左上角的点取两个box的max，交集的右下角的点取两个box的min
    #int_ymin = tf.maximum(bboxes1[...,0], bboxes2[...,0])
    #int_xmin = tf.maximum(bboxes1[...,1], bboxes2[...,1])
    #int_ymax = tf.minimum(bboxes1[...,2], bboxes2[...,2])
    #int_xmax = tf.minimum(bboxes1[...,3], bboxes2[...,3])

    #print('iou计算！ 1')
    #计算两个box的交集wh：如果两个box没有交集，那么wh为0（按照公式方式计算wh为负数，跟0比较取最大值）
    #int_h = tf.maximum(int_ymax - int_ymin, 0.)
    #int_w = tf.maximum(int_xmax - int_xmin, 0.)
    #print('iou计算！ 2')
    #计算IOU
    #int_vol = int_w * int_h #交集面积
    #vol1 = (bboxes1[...,2] - bboxes1[...,0]) * (bboxes1[...,3] - bboxes1[...,1]) #bboxes1的面积
    #vol2 = (bboxes2[...,2] - bboxes2[...,0]) * (bboxes2[...,3] - bboxes2[...,1]) #bboxes2的面积
    #iou = int_vol / (vol1 + vol2 - int_vol)
    #print('iou计算完成！！',iou)
    #return iou
  
  #计算loss
  def loss_layer(self, predicts, labels, scope='loss_layer'): 
    '''
    计算预测和标签之间的损失函数    
    args：
      predicts：Yolo网络的输出 形状[None,1470]  
            0：7*7*20：表示预测类别   
            7*7*20:7*7*20 + 7*7*2:表示预测置信度，即预测的边界框与实际边界框之间的IOU
            7*7*20 + 7*7*2：1470：预测边界框 目标中心是相对于当前格子的，宽度和高度的开根号是相对当前整张图像的(归一化的)
      labels：标签值 形状[None,7,7,25]
            0:1：置信度，表示这个地方是否有目标
            1:5：目标边界框  目标中心，宽度和高度(没有归一化)
            5:25：目标的类别 
    '''
    #label为（(45,7,7,25)）,5个为盒子信息，(x,y,w,h,c) 后20个为类别
    with tf.variable_scope(scope):
      # 预测值
      # 前20是class-20
      predict_classes = tf.reshape(
          predicts[:,:7*7*20],
          [self.batch_size, 7, 7, 20]
      )
      #confidence-2
      predict_confidence = tf.reshape(
          predicts[:, 7*7*20 : 7*7*20 + 7*7*2],
          [self.batch_size, 7, 7, 2]
      )
      # bounding box-2*4
      predict_boxes = tf.reshape(
          predicts[:, 7*7*20+7*7*2 :],
          [self.batch_size, 7, 7, 2, 4]
      )
      # 实际值 groundTruth
      #标签的置信度,表示这个地方是否有框 形状[45,7,7,1]
      # shape(45,7,7,1)
      # response中的值为0或者1，对应的网格中存在目标为1，不存在目标为0
      # 存在目标指的是存在目标的中心点，并不是说存在目标的一部分，所以，目标的中心所在的cell其对应的值才为1，其余的值均为0
      response = tf.reshape(
          labels[..., 0],
          [self.batch_size, 7, 7, 1]
      )
      # shape(45,7,7,1,4)
      # 标签的边界框 (x,y)表示边界框相对于整个图片的中心 形状[45,7,7,1，4]
      boxes = tf.reshape(
          labels[..., 1:5],
          [self.batch_size, 7, 7, 1, 4]
      )
      # shape(45,7,7,2,4),boxes的四个值，取值范围为0~1
      #标签的边界框 归一化后 张量沿着axis=3重复两边，扩充后[45,7,7,2,4]
      
      boxes = tf.tile(boxes, [1, 1, 1, 2, 1]) / 448#self.img_shape[0] #在这里由像素坐标变为0~1的比例值
      
      #shape(45, 7, 7, 20)
      classes = labels[..., 5:]

      # self.offset shape(7,7,2)
      # offset shape (1,7,7,2)      
      # shape (1, 7, 7, 2)
      x_offset = tf.reshape(
                tf.constant(self.x_offset, dtype=tf.float32),
                [1, 7, 7, 2])
      x_offset = tf.tile(x_offset, [self.batch_size, 1, 1, 1]) #(45, 7, 7, 2)
      #print('tile2!')
      # shape (45, 7, 7, 2)
      y_offset = tf.transpose(x_offset, (0, 2, 1, 3))

      # convert the x, y to the coordinates relative to the top left point of the image
      # the prediction of w, h are the square root
      # shape(45,7,7,2,4) -> (x,y,w,h)
      predicts_boxes_tran = tf.stack(
          [(predict_boxes[..., 0] + x_offset) / 7, #0~1,相对于整张图像的比例
           (predict_boxes[..., 1] + y_offset) / 7,
           tf.square(predict_boxes[..., 2]),
           tf.square(predict_boxes[..., 3])], axis=-1
      )

      #预测box与真实box的IOU，shape(45,7,7,2)
      iou_predict_truth = self.calc_iou(predicts_boxes_tran, boxes)

      # calculate I tensor [BATCH_SIZE, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
      # shape(45,7,7,1), find the maximum iou_predict_truth in every cell
      # 训练时，如果该单元中确实存在目标，那么只选择IOU最大的那个边界框来负责预测该目标，而其他的边界框认为不存在目标
      object_mask = tf.reduce_max(iou_predict_truth, axis=3, keep_dims=True) #最后一维，保留4阶tensor的尺寸
      # object probs (45,7,7,2)
      object_mask = tf.cast((iou_predict_truth >= object_mask), tf.float32) * response #cast是将bool转为float

      # calculate no_I tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
      # noobject confidence(45,7,7,2)
      # noobject_mask就表示每个边界框不负责该目标的置信度，
      # 使用tf.onr_like，使得全部为1,再减去有目标的，也就是有目标的对应坐标为1,这样一减，就变为没有的了。[45,7,7,2]
      noobject_probs = tf.ones_like(object_mask, dtype=tf.float32) - object_mask

      # shape(45,7,7,2,4), 对boxes的四个值进行规整，xy为相对于网格左上角，wh为取根号后的值，范围为0~1
      boxes_tran = tf.stack(
          [boxes[...,0] * 7 - x_offset,
           boxes[...,1] * 7 - y_offset,
           tf.sqrt(boxes[..., 2]),
           tf.sqrt(boxes[..., 3])], axis=-1
      )

      # class_loss shape(45,7,7,20)
      #class_loss 分类损失，如果目标出现在网格中 response为1，否则response为0  原文代价函数公式第5项
      #该项表名当格子中有目标时，预测的类别越接近实际类别，代价值越小  原文代价函数公式第5项
      class_delta = response * (predict_classes - classes) #[45,7,7,20]
      #tf.reduce_mean 函数用于计算张量tensor沿着指定的数轴（tensor的某一维度）上的的平均值，主要用作降维或者计算tensor（图像）的平均值。
      class_loss = tf.reduce_mean(tf.reduce_sum(tf.square(class_delta),axis=[1, 2, 3]), 
                     name='class_loss') * self.class_scale  #最后得到的就是一个标量数
      
      # object_loss confidence=iou*p(object)
      # p(object)的值为1或0
      object_delta = object_mask * (predict_confidence - iou_predict_truth)
      object_loss = tf.reduce_mean(
          tf.reduce_sum(
              tf.square(object_delta), axis=[1, 2, 3]
          ), name = 'object_loss'
      ) * self.object_scale

      # noobject_loss p(object)为0
      noobject_delta = noobject_probs * predict_confidence
      noobject_loss = tf.reduce_mean(
          tf.reduce_sum(
              tf.square(noobject_delta), axis=[1, 2, 3]
          ), name = 'noobject_loss'
      ) * self.noobject_scale

      # coord_loss
      coord_mask = tf.expand_dims(object_mask, 4)
      boxes_delta = coord_mask * (predict_boxes - boxes_tran)
      coord_loss = tf.reduce_mean(
          tf.reduce_sum(
              tf.square(boxes_delta), axis=[1,2,3,4]
          ), name='coord_loss'
      ) * self.coord_scale
      
      #将所有损失放在一起
      tf.losses.add_loss(class_loss)
      tf.losses.add_loss(object_loss)
      tf.losses.add_loss(noobject_loss)
      tf.losses.add_loss(coord_loss)

      self.class_loss = class_loss
      self.object_loss = object_loss
      self.noobject_loss = noobject_loss
      self.coord_loss = coord_loss

      # 将每个损失添加到日志记录
      tf.summary.scalar('class_loss', class_loss)
      tf.summary.scalar('object_loss', object_loss)
      tf.summary.scalar('noobject_loss', noobject_loss)
      tf.summary.scalar('coord_loss', coord_loss)

      tf.summary.histogram('boxes_delta_x', boxes_delta[..., 0])
      tf.summary.histogram('boxes_delta_y', boxes_delta[..., 1])
      tf.summary.histogram('boxes_delta_w', boxes_delta[..., 2])
      tf.summary.histogram('boxes_delta_h', boxes_delta[..., 3])
      tf.summary.histogram('iou', iou_predict_truth)
      #total_loss = class_loss + object_loss + noobject_loss + coord_loss
      #print('total_loss: ', total_loss)
      #return total_loss
      #return class_loss + object_loss + noobject_loss + coord_loss
    

  def train_yolo(self):
    global_step = tf.train.create_global_step()
    learning_rate = tf.train.exponential_decay(
        0.0001,
        global_step,
        30000,
        0.1,
        True,
        name='learning_rate'
    )
    op = tf.train.GradientDescentOptimizer(learning_rate).minimize()
