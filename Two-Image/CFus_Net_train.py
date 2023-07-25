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
input_C_1 = tf.placeholder(tf.float32, [None, None, None, 3], name='input_C_1')
input_C_2 = tf.placeholder(tf.float32, [None, None, None, 3], name='input_C_2')



input_1_hsv = tf.image.rgb_to_hsv(input_1)
input_1_h = tf.expand_dims(input_1_hsv[:,:,:,0],-1)
input_1_s = tf.expand_dims(input_1_hsv[:,:,:,1],-1)
input_1_v = tf.expand_dims(input_1_hsv[:,:,:,2],-1)

input_2_hsv = tf.image.rgb_to_hsv(input_2)
input_2_h = tf.expand_dims(input_2_hsv[:,:,:,0],-1)
input_2_s = tf.expand_dims(input_2_hsv[:,:,:,1],-1)
input_2_v = tf.expand_dims(input_2_hsv[:,:,:,2],-1)


[R_decom_1, C_decom_1, I_decom_1] = DecomNet(input_1,training=False)
[R_decom_2, C_decom_2, I_decom_2] = DecomNet(input_2,training=False)
#the output of decomposition network
decom_output_C1 = C_decom_1
decom_output_C2 = C_decom_2

#the output of CFuseNet 
Fus_output_C = CFusNet(input_C_1, input_C_2)

# color vector
Fuse_C_sum=tf.reduce_sum(Fus_output_C,axis=3,keepdims=True)
Fuse_C_vec=tf.div((Fus_output_C),(Fuse_C_sum+0.0001))

input_C_1_sum=tf.reduce_sum(input_C_1,axis=3,keepdims=True)
input_C_1_vec=tf.div((input_C_1),(input_C_1_sum+0.0001))

input_C_2_sum=tf.reduce_sum(input_C_2,axis=3,keepdims=True)
input_C_2_vec=tf.div((input_C_2),(input_C_2_sum+0.0001))


# mask decision block
mask_input_grad_1_x=tf.abs(gradient_no_norm(low_pass(R_decom_1), "x"))
mask_input_grad_1_y=tf.abs(gradient_no_norm(low_pass(R_decom_1), "y"))
mask_input_grad_1=mask_input_grad_1_x+mask_input_grad_1_y

mask_input_grad_2_x=tf.abs(gradient_no_norm(low_pass(R_decom_2), "x"))
mask_input_grad_2_y=tf.abs(gradient_no_norm(low_pass(R_decom_2), "y"))
mask_input_grad_2=mask_input_grad_2_x+mask_input_grad_2_y

# decision patch score for S Fusion
input_1_mask=1-tf.sign(tf.maximum(mask_input_grad_1,mask_input_grad_2)-mask_input_grad_1)
input_2_mask=1-input_1_mask


# patch decision block
score_input_grad_1_x=tf.abs(gradient_no_norm(low_pass(input_1_v), "x"))
score_input_grad_1_y=tf.abs(gradient_no_norm(low_pass(input_1_v), "y"))
score_input_grad_1=score_input_grad_1_x + score_input_grad_1_y

score_input_grad_2_x=tf.abs(gradient_no_norm(low_pass(input_2_v), "x"))
score_input_grad_2_y=tf.abs(gradient_no_norm(low_pass(input_2_v), "y"))
score_input_grad_2=score_input_grad_2_x+score_input_grad_2_y


input_patch_score = tf.nn.softmax(tf.concat([tf.expand_dims(tf.reduce_mean(score_input_grad_1, axis = [1,2,3])/0.1, axis = -1), tf.expand_dims(tf.reduce_mean(score_input_grad_2,axis = [1,2,3])/0.1, axis = -1)], axis = -1), axis = -1)
input_1_patch_score = input_patch_score[:,0:1]
input_2_patch_score = input_patch_score[:,1:2]




#define loss

#Fusion loss
#Intensity loss
NCV_loss = tf.reduce_mean(input_1_mask*tf.abs(Fuse_C_vec - input_C_1_vec)) + tf.reduce_mean(input_2_mask*tf.abs(Fuse_C_vec - input_C_2_vec))

#SSIM Fusion loss
S_SSIM_loss_1 = input_1_patch_score*(1-tf_ssim(Fus_output_C[:,:,:,0:1], input_C_1[:,:,:,0:1]))+input_2_patch_score*(1-tf_ssim(Fus_output_C[:,:,:,0:1], input_C_2[:,:,:,0:1]))
S_SSIM_loss_2 = input_1_patch_score*(1-tf_ssim(Fus_output_C[:,:,:,1:2], input_C_1[:,:,:,1:2]))+input_2_patch_score*(1-tf_ssim(Fus_output_C[:,:,:,1:2], input_C_2[:,:,:,1:2]))
S_SSIM_loss_3 = input_1_patch_score*(1-tf_ssim(Fus_output_C[:,:,:,2:3], input_C_1[:,:,:,2:3]))+input_2_patch_score*(1-tf_ssim(Fus_output_C[:,:,:,2:3], input_C_2[:,:,:,2:3]))
S_SSIM_loss=tf.reduce_mean(S_SSIM_loss_1+S_SSIM_loss_2+S_SSIM_loss_3)/3.0

#Fusion total loss
Cfus_loss_total =  5*NCV_loss + 1*S_SSIM_loss

with tf.name_scope('scalar'):
  tf.summary.scalar('NCV_loss',NCV_loss)
  tf.summary.scalar('S_SSIM_loss',S_SSIM_loss)
  tf.summary.scalar('Cfus_loss_total',Cfus_loss_total)
  tf.summary.scalar('input_1_patch_score',tf.reduce_mean(input_1_patch_score))
  tf.summary.scalar('input_2_patch_score',tf.reduce_mean(input_2_patch_score))



with tf.name_scope('image'):
  tf.summary.image('input_1',tf.expand_dims(input_1[1,:,:,:],0))
  tf.summary.image('input_2',tf.expand_dims(input_2[1,:,:,:],0))
  tf.summary.image('C_decom_1',tf.expand_dims(C_decom_1[1,:,:,:],0))  
  tf.summary.image('C_decom_2',tf.expand_dims(C_decom_2[1,:,:,:],0)) 
  tf.summary.image('Fus_output_C',tf.expand_dims(Fus_output_C[1,:,:,:],0))
  tf.summary.image('input_1_mask',tf.expand_dims(input_1_mask[1,:,:,:],0))
  tf.summary.image('input_2_mask',tf.expand_dims(input_2_mask[1,:,:,:],0))

        
summary_op = tf.summary.merge_all()
train_writer = tf.summary.FileWriter('./log/' + '/CFus_train',sess.graph,flush_secs=60)


lr = tf.placeholder(tf.float32, name='learning_rate')

optimizer = tf.train.AdamOptimizer(learning_rate=lr, name='AdamOptimizer')

var_Decom = [var for var in tf.trainable_variables() if 'DecomNet' in var.name]
var_CFus = [var for var in tf.trainable_variables() if 'CFusNet' in var.name]

saver_CFus = tf.train.Saver(var_list=var_CFus)
saver_Decom = tf.train.Saver(var_list = var_Decom)
train_op_CFus = optimizer.minimize(Cfus_loss_total, var_list = var_CFus)
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


decomposed_1_C_data = []
decomposed_2_C_data = []

input_1_img_data = []
input_2_img_data = []


for idx in range(len(train_1_data)):
    input_img1 = np.expand_dims(train_1_data[idx], axis=0)    
    CC1 = sess.run([decom_output_C1], feed_dict={input_1: input_img1})
    decom_output_C1_component = np.squeeze(CC1)
    input_img1_component = np.squeeze(input_img1)
    decomposed_1_C_data.append(decom_output_C1_component)
    input_1_img_data.append(input_img1_component)


    
for idx in range(len(train_2_data)):
    input_img2 = np.expand_dims(train_2_data[idx], axis=0)
    CC2= sess.run([decom_output_C2], feed_dict={input_2: input_img2})
    decom_output_C2_component = np.squeeze(CC2)
    input_img2_component = np.squeeze(input_img2)
    decomposed_2_C_data.append(decom_output_C2_component)
    input_2_img_data.append(input_img2_component)


train_CFuse_1_C_data = decomposed_1_C_data
train_CFuse_2_C_data = decomposed_2_C_data
train_Decom_1_img_data = input_1_img_data
train_Decom_2_img_data = input_2_img_data

print('[*] Number of training data: %d' % len(train_CFuse_1_C_data))

learning_rate = 0.0001
epoch = 2000
train_phase = 'CFuse'
numBatch = len(train_CFuse_1_C_data) // int(batch_size)
train_op = train_op_CFus
train_loss = Cfus_loss_total
saver = saver_CFus

checkpoint_dir = './checkpoint/CFus/'
if not os.path.isdir(checkpoint_dir):
    os.makedirs(checkpoint_dir)
ckpt=tf.train.get_checkpoint_state(checkpoint_dir)
if ckpt:
    print('loaded '+ckpt.model_checkpoint_path)
    saver.restore(sess,ckpt.model_checkpoint_path)
else:
    print("No CFuse Net pre model!")

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
        batch_input_1_C = np.zeros((batch_size, patch_size, patch_size, 3), dtype="float32")
        batch_input_2_C = np.zeros((batch_size, patch_size, patch_size, 3), dtype="float32")



        for patch_id in range(batch_size):
            C_1_data = train_CFuse_1_C_data[image_id]
            C_2_data = train_CFuse_2_C_data[image_id]

            
            img_1_data = train_Decom_1_img_data[image_id]
            img_2_data = train_Decom_2_img_data[image_id]
            
            

            h, w,_ = train_CFuse_1_C_data[image_id].shape
            x = random.randint(0, h - patch_size)
            y = random.randint(0, w - patch_size)
            C_1_data_crop = C_1_data[x : x+patch_size, y : y+patch_size, :]
            C_2_data_crop = C_2_data[x : x+patch_size, y : y+patch_size, :]
            img_1_data_crop = img_1_data[x : x+patch_size, y : y+patch_size, :]
            img_2_data_crop = img_2_data[x : x+patch_size, y : y+patch_size, :]
            
            rand_mode = np.random.randint(0, 7)
            batch_input_1_C[patch_id, :, :, :] = data_augmentation(C_1_data_crop , rand_mode)
            batch_input_2_C[patch_id, :, :, :] = data_augmentation(C_2_data_crop, rand_mode)
            batch_input_1_img[patch_id, :, :, :] = data_augmentation(img_1_data_crop, rand_mode)
            batch_input_2_img[patch_id, :, :, :] = data_augmentation(img_2_data_crop, rand_mode)


            image_id = (image_id + 1) % len(train_CFuse_1_C_data)
        counter += 1
        _, loss,summary_str = sess.run([train_op, train_loss,summary_op], feed_dict={input_1: batch_input_1_img, input_2: batch_input_2_img,input_C_1: batch_input_1_C,input_C_2: batch_input_2_C, lr: learning_rate})
        print("%s Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.6f" \
              % (train_phase, epoch + 1, batch_id + 1, numBatch, time.time() - start_time, loss))
        train_writer.add_summary(summary_str,counter)
        iter_num += 1
    if (epoch+1) % 2 ==0:       
      saver_CFus.save(sess, checkpoint_dir + 'model.ckpt', global_step=epoch+1)    


print("[*] Finish training for phase %s." % train_phase)



