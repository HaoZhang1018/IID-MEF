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
parser = argparse.ArgumentParser(description='')


parser.add_argument('--save_RC_dir', dest='save_RC_dir', default='./Results/IID/C/', help='directory for testing outputs')
parser.add_argument('--save_S_dir', dest='save_S_dir', default='./Results/IID/S/', help='directory for testing outputs')
parser.add_argument('--save_RT_dir', dest='save_RT_dir', default='./Results/IID/R/', help='directory for testing outputs')
parser.add_argument('--save_REC_dir', dest='save_REC_dir', default='./Results/IID/REC/', help='directory for testing outputs')
parser.add_argument('--test_dir', dest='test_dir', default='./dataset/test/demo/low/', help='directory for testing inputs')


args = parser.parse_args()

sess = tf.Session()
training = tf.placeholder_with_default(False, shape=(), name='training')
input_decom = tf.placeholder(tf.float32, [None, None, None, 3], name='input_decom')

[RT_decom, RC_decom, I_decom] = DecomNet(input_decom,training=False)
decom_output_RT = RT_decom
decom_output_RC = RC_decom
decom_output_I = I_decom




# load pretrained model
var_Decom = [var for var in tf.trainable_variables() if 'DecomNet' in var.name]
g_list = tf.global_variables()

saver_Decom = tf.train.Saver(var_list = var_Decom)


decom_checkpoint_dir ='./checkpoint/IID/'
ckpt_pre=tf.train.get_checkpoint_state(decom_checkpoint_dir)
if ckpt_pre:
    print('loaded '+ckpt_pre.model_checkpoint_path)
    saver_Decom.restore(sess,ckpt_pre.model_checkpoint_path)
else:
    print('No IID checkpoint!')

save_RC_dir = args.save_RC_dir
if not os.path.isdir(save_RC_dir):
    os.makedirs(save_RC_dir)

save_S_dir = args.save_S_dir
if not os.path.isdir(save_S_dir):
    os.makedirs(save_S_dir)    

save_RT_dir = args.save_RT_dir
if not os.path.isdir(save_RT_dir):
    os.makedirs(save_RT_dir) 

save_REC_dir = args.save_REC_dir
if not os.path.isdir(save_REC_dir):
    os.makedirs(save_REC_dir) 
        
###load eval data
eval_low_data = []
eval_img_name =[]
eval_low_data_name = glob(args.test_dir+'*')
eval_low_data_name.sort()
for idx in range(len(eval_low_data_name)):
    [_, name] = os.path.split(eval_low_data_name[idx])
    suffix = name[name.find('.') + 1:]
    name = name[:name.find('.')]
    eval_img_name.append(name)
    eval_low_im = load_images_no_norm(eval_low_data_name[idx])
    print(eval_low_im.shape)
    h,w,c = eval_low_im.shape
    h_tmp = h%1
    w_tmp = w%1
    eval_low_im_resize = eval_low_im[0:h-h_tmp, 0:w-w_tmp, :]
    print(eval_low_im_resize.shape)
    eval_low_data.append(eval_low_im_resize)


print("Start evalating!")
start_time = time.time()
for idx in range(len(eval_low_data)):
    print(idx)
    name = eval_img_name[idx]
    input_low = eval_low_data[idx]
    input_low_eval = np.expand_dims(input_low, axis=0)
    h, w, _ = input_low.shape

    decom_rt_low, decom_rc_low, decom_i_low = sess.run([decom_output_RT, decom_output_RC, decom_output_I], feed_dict={input_decom: input_low_eval})
    Rec_image=decom_rt_low*decom_rc_low*decom_i_low
    save_images(os.path.join(save_RT_dir, '%s.png' % (name)), decom_rt_low)
    save_images(os.path.join(save_RC_dir, '%s.png' % (name)), decom_rc_low)
    save_images_S(os.path.join(save_S_dir, '%s.png' % (name)), decom_i_low)
    save_images(os.path.join(save_REC_dir, '%s.png' % (name)), Rec_image)


    
    
