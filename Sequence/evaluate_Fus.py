# coding: utf-8
from __future__ import print_function
import os
import time
import random
from PIL import Image
import tensorflow as tf
import numpy as np
from utils import *
from model import *
from glob import glob
from skimage import color,filters
import argparse
import scipy.io as scio

parser = argparse.ArgumentParser(description='')

parser.add_argument('--save_RFus_dir', dest='save_RFus_dir', default='./Results/Fusion/RFus_result/', help='directory for testing outputs')
parser.add_argument('--save_SFus_dir', dest='save_SFus_dir', default='./Results/Fusion/SFus_result/', help='directory for testing outputs')
parser.add_argument('--save_CFus_dir', dest='save_CFus_dir', default='./Results/Fusion/CFus_result/', help='directory for testing outputs')
parser.add_argument('--save_IFus_dir', dest='save_IFus_dir', default='./Results/Fusion/Fus_image/', help='directory for testing outputs')
parser.add_argument('--test_1_dir', dest='test_1_dir', default='./dataset/test/demo/input_1/', help='directory for low inputs')
parser.add_argument('--test_2_dir', dest='test_2_dir', default='./dataset/test/demo/input_2/', help='directory for high inputs')
parser.add_argument('--test_3_dir', dest='test_3_dir', default='./dataset/test/demo/input_3/', help='directory for high inputs')




args = parser.parse_args()

sess = tf.Session()
training = tf.placeholder_with_default(False, shape=(), name='training')
input_1_image = tf.placeholder(tf.float32, [None, None, None, 3], name='input_1')
input_2_image = tf.placeholder(tf.float32, [None, None, None, 3], name='input_2')
input_3_image = tf.placeholder(tf.float32, [None, None, None, 3], name='input_3')

[R_1, C_1, S_1] = DecomNet(input_1_image)
[R_2, C_2, S_2] = DecomNet(input_2_image)
[R_3, C_3, S_3] = DecomNet(input_3_image)

Fus_R = RFusNet(R_1,R_2,R_3,training)
Fus_S = SFusNet(S_1,S_2,S_3,training)
Fus_C = CFusNet(C_1,C_2,C_3,training)
Fus_image=Fus_R*Fus_C*Fus_S

# load pretrained model
var_Decom = [var for var in tf.trainable_variables() if 'DecomNet' in var.name]
var_RFus =  [var for var in tf.trainable_variables() if 'RFusNet' in var.name]
var_SFus =  [var for var in tf.trainable_variables() if 'SFusNet' in var.name]
var_CFus =  [var for var in tf.trainable_variables() if 'CFusNet' in var.name]

g_list = tf.global_variables()

saver_Decom = tf.train.Saver(var_list = var_Decom)
saver_RFus = tf.train.Saver(var_list = var_RFus)
saver_SFus = tf.train.Saver(var_list = var_SFus)
saver_CFus = tf.train.Saver(var_list = var_CFus)


Decom_checkpoint_dir ='./checkpoint/IID/'
ckpt_pre=tf.train.get_checkpoint_state(Decom_checkpoint_dir)
if ckpt_pre:
    print('loaded '+ckpt_pre.model_checkpoint_path)
    saver_Decom.restore(sess,ckpt_pre.model_checkpoint_path)
else:
    print('No decomnet checkpoint!') 

RFus_checkpoint_dir ='./checkpoint/RFus/'
ckpt_pre_RFus=tf.train.get_checkpoint_state(RFus_checkpoint_dir)
if ckpt_pre_RFus:
    print('loaded '+ckpt_pre_RFus.model_checkpoint_path)
    saver_RFus.restore(sess,ckpt_pre_RFus.model_checkpoint_path)
else:
    print('No RFusNet checkpoint!')


SFus_checkpoint_dir ='./checkpoint/SFus/'
ckpt_pre_SFus=tf.train.get_checkpoint_state(SFus_checkpoint_dir)
if ckpt_pre_SFus:
    print('loaded '+ckpt_pre_SFus.model_checkpoint_path)
    saver_SFus.restore(sess,ckpt_pre_SFus.model_checkpoint_path)
else:
    print('No SFusNet checkpoint!')

CFus_checkpoint_dir ='./checkpoint/CFus/'
ckpt_pre_CFus=tf.train.get_checkpoint_state(CFus_checkpoint_dir)
if ckpt_pre_CFus:
    print('loaded '+ckpt_pre_CFus.model_checkpoint_path)
    saver_CFus.restore(sess,ckpt_pre_CFus.model_checkpoint_path)
else:
    print('No CFusNet checkpoint!')


save_RFus_dir = args.save_RFus_dir
if not os.path.isdir(save_RFus_dir):
    os.makedirs(save_RFus_dir)
    
save_SFus_dir = args.save_SFus_dir
if not os.path.isdir(save_SFus_dir):
    os.makedirs(save_SFus_dir)
    
save_IFus_dir = args.save_IFus_dir
if not os.path.isdir(save_IFus_dir):
    os.makedirs(save_IFus_dir)    

save_CFus_dir = args.save_CFus_dir
if not os.path.isdir(save_CFus_dir):
    os.makedirs(save_CFus_dir)  
    
###load eval data
eval_1_data = []
eval_1_img_name =[]
eval_2_data = []
eval_2_img_name =[]
eval_3_data = []
eval_3_img_name =[]


eval_1_data_name = glob(args.test_1_dir+'*') 
eval_1_data_name.sort() 
eval_2_data_name = glob(args.test_2_dir+'*') 
eval_2_data_name.sort() 
eval_3_data_name = glob(args.test_3_dir+'*') 
eval_3_data_name.sort() 


for idx in range(len(eval_1_data_name)):
    [_, name_1] = os.path.split(eval_1_data_name[idx])
    [_, name_2] = os.path.split(eval_2_data_name[idx])
    [_, name_3] = os.path.split(eval_3_data_name[idx])
        
    suffix_1 = name_1[name_1.find('.') + 1:]
    name_1 = name_1[:name_1.find('.')]
    suffix_2 = name_2[name_2.find('.') + 1:]
    name_2 = name_2[:name_2.find('.')]   
    suffix_3 = name_3[name_3.find('.') + 1:]
    name_3 = name_3[:name_3.find('.')] 
    
    
        
    eval_1_img_name.append(name_1)
    eval_1_im = load_images_no_norm(eval_1_data_name[idx])
    eval_2_img_name.append(name_2)
    eval_2_im = load_images_no_norm(eval_2_data_name[idx])
    eval_3_img_name.append(name_3)
    eval_3_im = load_images_no_norm(eval_3_data_name[idx])        

    eval_1_data.append(eval_1_im)
    eval_2_data.append(eval_2_im)
    eval_3_data.append(eval_3_im)


print("Start evalating!")


for idx in range(len(eval_1_data)):
    print(idx)
    name_1 = eval_1_img_name[idx]
    name_2 = eval_2_img_name[idx]
    name_3 = eval_3_img_name[idx]
    
    input_1 = eval_1_data[idx]
    input_2 = eval_2_data[idx]
    input_3 = eval_3_data[idx]
    
    input_1_eval = np.expand_dims(input_1, axis=0)
    input_2_eval = np.expand_dims(input_2, axis=0)
    input_3_eval = np.expand_dims(input_3, axis=0)
    
    h, w, _ = input_1.shape
    Re_Fusion,Sh_Fusion, Co_Fusion,Img_Fusion = sess.run([Fus_R,Fus_S,Fus_C,Fus_image], feed_dict={input_1_image:input_1_eval,input_2_image:input_2_eval,input_3_image:input_3_eval,training:False})
              
    save_images(os.path.join(save_RFus_dir, '%s.png' % (name_1)), Re_Fusion)
    save_images_S(os.path.join(save_SFus_dir, '%s.png' % (name_1)), Sh_Fusion)
    save_images(os.path.join(save_CFus_dir, '%s.png' % (name_1)), Co_Fusion)
    save_images(os.path.join(save_IFus_dir, '%s.png' % (name_1)), Img_Fusion)

    
    
