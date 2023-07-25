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
input_R_1 = tf.placeholder(tf.float32, [None, None, None, 1], name='input_R_1')
input_R_2 = tf.placeholder(tf.float32, [None, None, None, 1], name='input_R_2')


input_1_hsv = tf.image.rgb_to_hsv(input_1)
input_1_h = tf.expand_dims(input_1_hsv[:,:,:,0],-1)
input_1_v = tf.expand_dims(input_1_hsv[:,:,:,2],-1)

input_2_hsv = tf.image.rgb_to_hsv(input_2)
input_2_h = tf.expand_dims(input_2_hsv[:,:,:,0],-1)
input_2_v = tf.expand_dims(input_2_hsv[:,:,:,2],-1)




[R_decom_1, C_decom_1, I_decom_1] = DecomNet(input_1,training=False)
[R_decom_2, C_decom_2, I_decom_2] = DecomNet(input_2,training=False)
#the output of decomposition network
decom_output_R1 = R_decom_1
decom_output_R2 = R_decom_2

#the output of RFuseNet 
Fus_output_R = RFusNet(input_R_1, input_R_2)

#define loss

input_grad_1_x=tf.abs(gradient_no_norm(low_pass(input_1_v), "x"))
input_grad_1_y=tf.abs(gradient_no_norm(low_pass(input_1_v), "y"))
input_grad_1=input_grad_1_x+input_grad_1_y

input_2_gray=tf.image.rgb_to_grayscale(input_2)
input_grad_2_x=tf.abs(gradient_no_norm(low_pass(input_2_v), "x"))
input_grad_2_y=tf.abs(gradient_no_norm(low_pass(input_2_v), "y"))
input_grad_2=input_grad_2_x+input_grad_2_y

input_1_mask=1-tf.sign(tf.maximum(input_grad_1,input_grad_2)-input_grad_1)
input_2_mask=1-input_1_mask


#gradient loss
Gra_loss = tf.reduce_mean(input_1_mask*tf.abs(gradient_no_norm_no_abs(Fus_output_R,"x")-gradient_no_norm_no_abs(input_R_1,"x")))+tf.reduce_mean(input_1_mask*tf.abs(gradient_no_norm_no_abs(Fus_output_R,"y")-gradient_no_norm_no_abs(input_R_1,"y")))+ tf.reduce_mean(input_2_mask*tf.abs(gradient_no_norm_no_abs(Fus_output_R,"x")-gradient_no_norm_no_abs(input_R_2,"x")))+tf.reduce_mean(input_2_mask*tf.abs(gradient_no_norm_no_abs(Fus_output_R,"y")-gradient_no_norm_no_abs(input_R_2,"y")))

#Fusion total loss
Rfus_loss_total =  1*Gra_loss 

tf.summary.scalar('Gra_loss',Gra_loss)
tf.summary.scalar('Rfus_loss_total',Rfus_loss_total)

with tf.name_scope('image'):
  tf.summary.image('input_1',tf.expand_dims(input_1[1,:,:,:],0))
  tf.summary.image('input_2',tf.expand_dims(input_2[1,:,:,:],0))
  tf.summary.image('input_1_v',tf.expand_dims(input_1_v[1,:,:,:],0))
  tf.summary.image('input_2_v',tf.expand_dims(input_2_v[1,:,:,:],0))
  tf.summary.image('R_decom_1',tf.expand_dims(R_decom_1[1,:,:,:],0))  
  tf.summary.image('R_decom_2',tf.expand_dims(R_decom_2[1,:,:,:],0)) 
  tf.summary.image('Fus_output_R',tf.expand_dims(Fus_output_R[1,:,:,:],0))
  tf.summary.image('input_2_mask',tf.expand_dims(input_2_mask[1,:,:,:],0)) 

          
summary_op = tf.summary.merge_all()
train_writer = tf.summary.FileWriter('./log/' + '/RFus_train',sess.graph,flush_secs=60)


lr = tf.placeholder(tf.float32, name='learning_rate')

optimizer = tf.train.AdamOptimizer(learning_rate=lr, name='AdamOptimizer')

var_Decom = [var for var in tf.trainable_variables() if 'DecomNet' in var.name]
var_RFus = [var for var in tf.trainable_variables() if 'RFusNet' in var.name]

saver_RFus = tf.train.Saver(var_list=var_RFus)
saver_Decom = tf.train.Saver(var_list = var_Decom)
train_op_RFus = optimizer.minimize(Rfus_loss_total, var_list = var_RFus)
sess.run(tf.global_variables_initializer())
print("[*] Initialize model successfully...")

### load data
train_1_data = []
train_2_data = []

train_1_data_names = glob(args.train_data_dir +'/low/*')
train_1_data_names.sort()

train_2_data_names = glob(args.train_data_dir +'/high/*') 
train_2_data_names.sort()

assert len(train_1_data_names) == len(train_2_data_names)
print('[*] Number of training data: %d' % len(train_1_data_names))

for idx in range(len(train_1_data_names)):
    im_1 = load_images_no_norm(train_1_data_names[idx])
    train_1_data.append(im_1)
    im_2 = load_images_no_norm(train_2_data_names[idx])
    train_2_data.append(im_2)

pre_decom_checkpoint_dir = './checkpoint/IID/'
ckpt_pre=tf.train.get_checkpoint_state(pre_decom_checkpoint_dir)
if ckpt_pre:
    print('loaded '+ckpt_pre.model_checkpoint_path)
    saver_Decom.restore(sess,ckpt_pre.model_checkpoint_path)
else:
    print('No pre_decom_net checkpoint!')


decomposed_1_R_data = []
decomposed_2_R_data = []

input_1_img_data = []
input_2_img_data = []


for idx in range(len(train_1_data)):
    input_img1 = np.expand_dims(train_1_data[idx], axis=0)    
    RR1 = sess.run([decom_output_R1], feed_dict={input_1: input_img1})
    decom_output_R1_component = np.squeeze(RR1)
    input_img1_component = np.squeeze(input_img1)
    print(decom_output_R1_component.shape)
    decomposed_1_R_data.append(decom_output_R1_component)
    input_1_img_data.append(input_img1_component)


    
for idx in range(len(train_2_data)):
    input_img2 = np.expand_dims(train_2_data[idx], axis=0)
    RR2= sess.run([decom_output_R2], feed_dict={input_2: input_img2})
    decom_output_R2_component = np.squeeze(RR2)
    input_img2_component = np.squeeze(input_img2)
    print(decom_output_R2_component.shape)
    decomposed_2_R_data.append(decom_output_R2_component)
    input_2_img_data.append(input_img2_component)


train_RFuse_1_R_data = decomposed_1_R_data
train_RFuse_2_R_data = decomposed_2_R_data
train_Decom_1_img_data = input_1_img_data
train_Decom_2_img_data = input_2_img_data



print('[*] Number of training data: %d' % len(train_RFuse_1_R_data))

learning_rate = 0.0001
epoch = 2000
train_phase = 'RFuse'
numBatch = len(train_RFuse_1_R_data) // int(batch_size)
train_op = train_op_RFus
train_loss = Rfus_loss_total
saver = saver_RFus

checkpoint_dir = './checkpoint/RFus/'
if not os.path.isdir(checkpoint_dir):
    os.makedirs(checkpoint_dir)
ckpt=tf.train.get_checkpoint_state(checkpoint_dir)
if ckpt:
    print('loaded '+ckpt.model_checkpoint_path)
    saver.restore(sess,ckpt.model_checkpoint_path)
else:
    print("No RFuse Net pre model!")

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
        batch_input_1_R = np.zeros((batch_size, patch_size, patch_size, 1), dtype="float32")
        batch_input_2_R = np.zeros((batch_size, patch_size, patch_size, 1), dtype="float32")



        for patch_id in range(batch_size):
            R_1_data = train_RFuse_1_R_data[image_id]
            R_1_expand = np.expand_dims(R_1_data, axis = 2)
            R_2_data = train_RFuse_2_R_data[image_id]
            R_2_expand = np.expand_dims(R_2_data, axis = 2)
            
            img_1_data = train_Decom_1_img_data[image_id]
            img_2_data = train_Decom_2_img_data[image_id]
            
            

            h, w = train_RFuse_1_R_data[image_id].shape
            x = random.randint(0, h - patch_size)
            y = random.randint(0, w - patch_size)
            R_1_data_crop = R_1_expand[x : x+patch_size, y : y+patch_size, :]
            R_2_data_crop = R_2_expand[x : x+patch_size, y : y+patch_size, :]
            img_1_data_crop = img_1_data[x : x+patch_size, y : y+patch_size, :]
            img_2_data_crop = img_2_data[x : x+patch_size, y : y+patch_size, :]
            
            rand_mode = np.random.randint(0, 7)
            batch_input_1_R[patch_id, :, :, :] = data_augmentation(R_1_data_crop , rand_mode)
            batch_input_2_R[patch_id, :, :, :] = data_augmentation(R_2_data_crop, rand_mode)
            batch_input_1_img[patch_id, :, :, :] = data_augmentation(img_1_data_crop, rand_mode)
            batch_input_2_img[patch_id, :, :, :] = data_augmentation(img_2_data_crop, rand_mode)


            image_id = (image_id + 1) % len(train_RFuse_1_R_data)
        counter += 1
        _, loss,summary_str = sess.run([train_op, train_loss,summary_op], feed_dict={input_1: batch_input_1_img, input_2: batch_input_2_img,input_R_1: batch_input_1_R,input_R_2: batch_input_2_R, lr: learning_rate})
        print("%s Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.6f" \
              % (train_phase, epoch + 1, batch_id + 1, numBatch, time.time() - start_time, loss))
        train_writer.add_summary(summary_str,counter)
        iter_num += 1
    if (epoch+1) % 2 ==0:       
      saver_RFus.save(sess, checkpoint_dir + 'model.ckpt', global_step=epoch+1)    


print("[*] Finish training for phase %s." % train_phase)



