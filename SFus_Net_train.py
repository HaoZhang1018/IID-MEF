# coding: utf-8
from __future__ import print_function
import os
import time
import random
#from skimage import color
from PIL import Image
import tensorflow as tf
import numpy as np
from utils import *
from model import *
from glob import glob
import argparse


parser = argparse.ArgumentParser(description='')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=10, help='number of samples in one batch')
parser.add_argument('--patch_size', dest='patch_size', type=int, default=48, help='patch size')
parser.add_argument('--train_data_dir', dest='train_data_dir', default='./dataset/train', help='directory for training inputs')

args = parser.parse_args()

batch_size = args.batch_size
patch_size = args.patch_size
sess = tf.Session()

input_1 = tf.placeholder(tf.float32, [None, None, None, 3], name='input_1')
input_2 = tf.placeholder(tf.float32, [None, None, None, 3], name='input_2')


#the input of illumination adjustment net
input_S_1 = tf.placeholder(tf.float32, [None, None, None, 1], name='input_S_1')
input_S_2 = tf.placeholder(tf.float32, [None, None, None, 1], name='input_S_2')


input_1_hsv = tf.image.rgb_to_hsv(input_1)
input_1_h = tf.expand_dims(input_1_hsv[:,:,:,0],-1)
input_1_v = tf.expand_dims(input_1_hsv[:,:,:,2],-1)

input_2_hsv = tf.image.rgb_to_hsv(input_2)
input_2_h = tf.expand_dims(input_2_hsv[:,:,:,0],-1)
input_2_v = tf.expand_dims(input_2_hsv[:,:,:,2],-1)

[R_decom_1, C_decom_1, S_decom_1] = DecomNet(input_1,training=False)
[R_decom_2, C_decom_2, S_decom_2] = DecomNet(input_2,training=False)


#the output of decomposition network
decom_output_S1 = S_decom_1
decom_output_S2 = S_decom_2

#the output of SFuseNet 
Fus_output_S = SFusNet(input_S_1, input_S_2)




score_input_grad_1_x=tf.abs(gradient_no_norm(low_pass(input_1_v), "x"))
score_input_grad_1_y=tf.abs(gradient_no_norm(low_pass(input_1_v), "y"))
score_input_grad_1=score_input_grad_1_x + score_input_grad_1_y

score_input_grad_2_x=tf.abs(gradient_no_norm(low_pass(input_2_v), "x"))
score_input_grad_2_y=tf.abs(gradient_no_norm(low_pass(input_2_v), "y"))
score_input_grad_2=score_input_grad_2_x+score_input_grad_2_y


input_patch_score = tf.nn.softmax(tf.concat([tf.expand_dims(tf.reduce_mean(score_input_grad_1, axis = [1,2,3])/0.05, axis = -1), tf.expand_dims(tf.reduce_mean(score_input_grad_2,axis = [1,2,3])/0.05, axis = -1)], axis = -1))
input_1_patch_score = input_patch_score[:,0:1]
input_2_patch_score = input_patch_score[:,1:2]



#define loss
#SSIM Fusion loss
S_SSIM_loss = tf.reduce_mean(input_1_patch_score*(1-tf_ssim(Fus_output_S, input_S_1))+input_2_patch_score*(1-tf_ssim(Fus_output_S, input_S_2)))

Sfus_loss_total = 1*S_SSIM_loss




with tf.name_scope('scalar'):
  tf.summary.scalar('Sfus_loss_total',Sfus_loss_total) 
  tf.summary.scalar('input_1_patch_score',tf.reduce_mean(input_1_patch_score))
  tf.summary.scalar('input_2_patch_score',tf.reduce_mean(input_2_patch_score))
  
  
with tf.name_scope('image'): 
  tf.summary.image('input_1',tf.expand_dims(input_1[1,:,:,:],0))
  tf.summary.image('input_2',tf.expand_dims(input_2[1,:,:,:],0))
  tf.summary.image('input_S_1',tf.expand_dims(input_S_1[1,:,:,:],0))
  tf.summary.image('input_S_2',tf.expand_dims(input_S_2[1,:,:,:],0))
  tf.summary.image('Fus_output_S',tf.expand_dims(Fus_output_S[1,:,:,:],0))

  
        
summary_op = tf.summary.merge_all()
train_writer = tf.summary.FileWriter('./log/' + '/SFus_train',sess.graph,flush_secs=60)


lr = tf.placeholder(tf.float32, name='learning_rate')

optimizer = tf.train.AdamOptimizer(learning_rate=lr, name='AdamOptimizer')

var_Decom = [var for var in tf.trainable_variables() if 'DecomNet' in var.name]
var_SFus = [var for var in tf.trainable_variables() if 'SFusNet' in var.name]

saver_SFus = tf.train.Saver(var_list=var_SFus)
saver_Decom = tf.train.Saver(var_list = var_Decom)
train_op_SFus = optimizer.minimize(Sfus_loss_total, var_list = var_SFus)
sess.run(tf.global_variables_initializer())
print("[*] Initialize model successfully...")

### load data
train_1_data = []
train_2_data = []

train_1_data_names = glob(args.train_data_dir +'/low/*.jpg')
train_1_data_names.sort()

train_2_data_names = glob(args.train_data_dir +'/high/*.jpg') 
train_2_data_names.sort()

assert len(train_1_data_names) == len(train_2_data_names)
print('[*] Number of training data: %d' % len(train_1_data_names))

for idx in range(len(train_1_data_names)):
    im_1 = load_images_no_norm(train_1_data_names[idx])
    train_1_data.append(im_1)
    im_2 = load_images_no_norm(train_2_data_names[idx])
    train_2_data.append(im_2)

pre_decom_checkpoint_dir = './checkpoint/decom_net_retrain/'
ckpt_pre=tf.train.get_checkpoint_state(pre_decom_checkpoint_dir)
if ckpt_pre:
    print('loaded '+ckpt_pre.model_checkpoint_path)
    saver_Decom.restore(sess,ckpt_pre.model_checkpoint_path)
else:
    print('No pre_decom_net checkpoint!')


decomposed_1_S_data = []
decomposed_2_S_data = []

input_1_img_data = []
input_2_img_data = []


for idx in range(len(train_1_data)):
    input_img1 = np.expand_dims(train_1_data[idx], axis=0)    
    SS1 = sess.run([decom_output_S1], feed_dict={input_1: input_img1})
    decom_output_S1_component = np.squeeze(SS1)
    input_img1_component = np.squeeze(input_img1)
    print(decom_output_S1_component.shape)
    decomposed_1_S_data.append(decom_output_S1_component)
    input_1_img_data.append(input_img1_component)


    
for idx in range(len(train_2_data)):
    input_img2 = np.expand_dims(train_2_data[idx], axis=0)
    SS2= sess.run([decom_output_S2], feed_dict={input_2: input_img2})
    decom_output_S2_component = np.squeeze(SS2)
    input_img2_component = np.squeeze(input_img2)
    print(decom_output_S2_component.shape)
    decomposed_2_S_data.append(decom_output_S2_component)
    input_2_img_data.append(input_img2_component)


train_SFuse_1_S_data = decomposed_1_S_data
train_SFuse_2_S_data = decomposed_2_S_data
train_Decom_1_img_data = input_1_img_data
train_Decom_2_img_data = input_2_img_data



print('[*] Number of training data: %d' % len(train_SFuse_1_S_data))

learning_rate = 0.0001
epoch = 2000
train_phase = 'SFuse'
numBatch = len(train_SFuse_1_S_data) // int(batch_size)
train_op = train_op_SFus
train_loss = Sfus_loss_total
saver = saver_SFus

checkpoint_dir = './checkpoint/SFus/'
if not os.path.isdir(checkpoint_dir):
    os.makedirs(checkpoint_dir)
ckpt=tf.train.get_checkpoint_state(checkpoint_dir)
if ckpt:
    print('loaded '+ckpt.model_checkpoint_path)
    saver.restore(sess,ckpt.model_checkpoint_path)
else:
    print("No SFuse Net pre model!")

start_step = 0
start_epoch = 0
iter_num = 0
print("[*] Start training for phase %s, with start epoch %d start iter %d : " % (train_phase, start_epoch, iter_num))

start_time = time.time()
image_id = 0
counter = 0
for epoch in range(start_epoch, epoch):
    for batch_id in range(start_step, numBatch):
        batch_input_1_img = np.zeros((batch_size, patch_size, patch_size, 3), dtype="float32")
        batch_input_2_img = np.zeros((batch_size, patch_size, patch_size, 3), dtype="float32")
        batch_input_1_S = np.zeros((batch_size, patch_size, patch_size, 1), dtype="float32")
        batch_input_2_S = np.zeros((batch_size, patch_size, patch_size, 1), dtype="float32")

        for patch_id in range(batch_size):
            S_1_data = train_SFuse_1_S_data[image_id]
            S_1_expand = np.expand_dims(S_1_data, axis = 2)
            S_2_data = train_SFuse_2_S_data[image_id]
            S_2_expand = np.expand_dims(S_2_data, axis = 2)
            
            img_1_data = train_Decom_1_img_data[image_id]
            img_2_data = train_Decom_2_img_data[image_id]
            
            

            h, w = train_SFuse_1_S_data[image_id].shape
            x = random.randint(0, h - patch_size)
            y = random.randint(0, w - patch_size)
            S_1_data_crop = S_1_expand[x : x+patch_size, y : y+patch_size, :]
            S_2_data_crop = S_2_expand[x : x+patch_size, y : y+patch_size, :]
            img_1_data_crop = img_1_data[x : x+patch_size, y : y+patch_size, :]
            img_2_data_crop = img_2_data[x : x+patch_size, y : y+patch_size, :]
            
            rand_mode = np.random.randint(0, 7)
            batch_input_1_S[patch_id, :, :, :] = data_augmentation(S_1_data_crop , rand_mode)
            batch_input_2_S[patch_id, :, :, :] = data_augmentation(S_2_data_crop, rand_mode)
            batch_input_1_img[patch_id, :, :, :] = data_augmentation(img_1_data_crop, rand_mode)
            batch_input_2_img[patch_id, :, :, :] = data_augmentation(img_2_data_crop, rand_mode)


            image_id = (image_id + 1) % len(train_SFuse_1_S_data)
        counter += 1
        _, loss,summary_str = sess.run([train_op, train_loss,summary_op], feed_dict={input_1: batch_input_1_img, input_2: batch_input_2_img,input_S_1: batch_input_1_S,input_S_2: batch_input_2_S, lr: learning_rate})
        print("%s Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.6f" \
              % (train_phase, epoch + 1, batch_id + 1, numBatch, time.time() - start_time, loss))
        train_writer.add_summary(summary_str,counter)
        iter_num += 1
    if (epoch+1) % 2 ==0:       
      saver_SFus.save(sess, checkpoint_dir + 'model.ckpt', global_step=epoch+1)    


print("[*] Finish training for phase %s." % train_phase)



