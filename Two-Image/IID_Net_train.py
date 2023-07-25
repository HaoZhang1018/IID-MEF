# coding: utf-8
from __future__ import print_function
import os, time, random
import tensorflow as tf
from PIL import Image
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

input_low = tf.placeholder(tf.float32, [None, None, None, 3], name='input_low')
input_high = tf.placeholder(tf.float32, [None, None, None, 3], name='input_high')

[R_low,  C_low,  I_low] = DecomNet(input_low)
[R_high, C_high, I_high] = DecomNet(input_high)

R_low_3 = tf.concat([R_low, R_low, R_low], axis=3)
R_high_3 = tf.concat([R_high, R_high, R_high], axis=3)
C_low_3 = C_low
C_high_3 = C_high
I_low_3 = tf.concat([I_low, I_low, I_low], axis=3)
I_high_3 = tf.concat([I_high, I_high, I_high], axis=3)


#network output
output_R_low = R_low_3
output_R_high = R_high_3
output_C_low = C_low_3
output_C_high = C_high_3
output_I_low = I_low_3
output_I_high = I_high_3


# color vector
input_low_sum=tf.reduce_sum(input_low,axis=3,keepdims=True)
input_low_vec=tf.div(input_low,tf.maximum(input_low_sum, 0.01))

input_high_sum=tf.reduce_sum(input_high,axis=3,keepdims=True)
input_high_vec=tf.div(input_high,tf.maximum(input_high_sum, 0.01))

output_C_low_sum=tf.reduce_sum(output_C_low,axis=3,keepdims=True)
output_C_low_vec=tf.div((output_C_low),tf.maximum(output_C_low_sum, 0.01))

output_C_high_sum=tf.reduce_sum(output_C_high,axis=3,keepdims=True)
output_C_high_vec=tf.div((output_C_high),tf.maximum(output_C_high_sum, 0.01))


input_low_hsv = tf.image.rgb_to_hsv(input_low)
input_low_h = tf.expand_dims(input_low_hsv[:,:,:,0],-1)
input_low_v = tf.expand_dims(input_low_hsv[:,:,:,2],-1)

input_high_hsv = tf.image.rgb_to_hsv(input_high)
input_high_h = tf.expand_dims(input_high_hsv[:,:,:,0],-1)
input_high_v = tf.expand_dims(input_high_hsv[:,:,:,2],-1)

output_C_low_hsv = tf.image.rgb_to_hsv(output_C_low)
output_C_low_h = tf.expand_dims(output_C_low_hsv[:,:,:,0],-1)
output_C_low_v = tf.expand_dims(output_C_low_hsv[:,:,:,2],-1)

output_C_high_hsv = tf.image.rgb_to_hsv(output_C_high)
output_C_high_h = tf.expand_dims(output_C_high_hsv[:,:,:,0],-1)
output_C_high_v = tf.expand_dims(output_C_high_hsv[:,:,:,2],-1)

# define loss
def mutual_i_input_loss(input_I_low, input_im):
    low_gradient_x = gradient_no_norm(input_I_low, "x")
    input_gradient_x = gradient_no_norm(input_im, "x")
    x_loss = tf.abs(tf.div(low_gradient_x, tf.maximum(input_gradient_x, 0.01)))
    low_gradient_y = gradient_no_norm(input_I_low, "y")
    input_gradient_y = gradient_no_norm(input_im, "y")
    y_loss = tf.abs(tf.div(low_gradient_y, tf.maximum(input_gradient_y, 0.01)))
    mut_loss = tf.reduce_mean(x_loss + y_loss) 
    return mut_loss


def mutual_C_input_loss(input_C_low_3, input_im_H):
    input_C_low_1=tf.expand_dims(input_C_low_3[:,:,:,0],-1)
    low_gradient_1_x = gradient_no_norm(input_C_low_1, "x")
    input_gradient_1_x = gradient_no_norm(input_im_H, "x")
    x_loss_1 = tf.abs(tf.div(low_gradient_1_x, tf.maximum(input_gradient_1_x, 0.01)))
    low_gradient_1_y = gradient_no_norm(input_C_low_1, "y")
    input_gradient_1_y = gradient_no_norm(input_im_H, "y")
    y_loss_1 = tf.abs(tf.div(low_gradient_1_y, tf.maximum(input_gradient_1_y, 0.01)))
    mut_loss_1 = tf.reduce_mean(x_loss_1 + y_loss_1) 
    
    input_C_low_2=tf.expand_dims(input_C_low_3[:,:,:,1],-1)
    low_gradient_2_x = gradient_no_norm(input_C_low_2, "x")
    input_gradient_2_x = gradient_no_norm(input_im_H, "x")
    x_loss_2 = tf.abs(tf.div(low_gradient_2_x, tf.maximum(input_gradient_2_x, 0.01)))
    low_gradient_2_y = gradient_no_norm(input_C_low_2, "y")
    input_gradient_2_y = gradient_no_norm(input_im_H, "y")
    y_loss_2 = tf.abs(tf.div(low_gradient_2_y, tf.maximum(input_gradient_2_y, 0.01)))
    mut_loss_2 = tf.reduce_mean(x_loss_2 + y_loss_2)    
    
    
    input_C_low_3=tf.expand_dims(input_C_low_3[:,:,:,2],-1)
    low_gradient_3_x = gradient_no_norm(input_C_low_3, "x")
    input_gradient_3_x = gradient_no_norm(input_im_H, "x")
    x_loss_3 = tf.abs(tf.div(low_gradient_3_x, tf.maximum(input_gradient_3_x, 0.01)))
    low_gradient_3_y = gradient_no_norm(input_C_low_3, "y")
    input_gradient_3_y = gradient_no_norm(input_im_H, "y")
    y_loss_3 = tf.abs(tf.div(low_gradient_3_y, tf.maximum(input_gradient_3_y, 0.01)))
    mut_loss_3 = tf.reduce_mean(x_loss_3 + y_loss_3)        
    
    mut_loss=(mut_loss_1+mut_loss_2+mut_loss_3)/3;
    return mut_loss




recon_loss_low = tf.reduce_mean(tf.abs(((output_R_low*output_C_low) * output_I_low)- input_low))
recon_loss_high = tf.reduce_mean(tf.abs(((output_R_high*output_C_high) * output_I_high)- input_high))
equal_R_loss=tf.reduce_mean(tf.abs(output_R_low-output_R_high))


i_input_mutual_loss_high = mutual_i_input_loss(I_high, input_high_v)
i_input_mutual_loss_low = mutual_i_input_loss(I_low, input_low_v)


C_input_mutual_loss_high = mutual_C_input_loss(output_C_high, input_high_h)
C_input_mutual_loss_low = mutual_C_input_loss(output_C_low, input_low_h)



loss_Decom = 1*recon_loss_high + 1*recon_loss_low + 0.1 * equal_R_loss \
              + 0.008* i_input_mutual_loss_high + 0.008* i_input_mutual_loss_low \
              + 0.008* C_input_mutual_loss_high + 0.008* C_input_mutual_loss_low


              
tf.summary.scalar('recon_loss_high',recon_loss_high)
tf.summary.scalar('recon_loss_low',recon_loss_low)
tf.summary.scalar('equal_R_loss',equal_R_loss)
tf.summary.scalar('i_input_mutual_loss_high',i_input_mutual_loss_high)
tf.summary.scalar('i_input_mutual_loss_low',i_input_mutual_loss_low)
tf.summary.scalar('C_input_mutual_loss_high',C_input_mutual_loss_high)
tf.summary.scalar('C_input_mutual_loss_low',C_input_mutual_loss_low)
tf.summary.scalar('loss_Decom',loss_Decom)

tf.summary.scalar('output_C_low_h',tf.reduce_mean(tf.abs(input_low_h)))
tf.summary.scalar('output_C_low_v',tf.reduce_mean(tf.abs(input_low_v)))
###
lr = tf.placeholder(tf.float32, name='learning_rate')

optimizer = tf.train.AdamOptimizer(learning_rate=lr, name='AdamOptimizer')
var_Decom = [var for var in tf.trainable_variables() if 'DecomNet' in var.name]
train_op_Decom = optimizer.minimize(loss_Decom, var_list = var_Decom)
sess.run(tf.global_variables_initializer())

saver_Decom = tf.train.Saver(var_list = var_Decom,max_to_keep=4000)
print("[*] Initialize model successfully...")

with tf.name_scope('image'):
  tf.summary.image('input_low',tf.expand_dims(input_low[1,:,:,:],0))
  tf.summary.image('input_high',tf.expand_dims(input_high[1,:,:,:],0))
  tf.summary.image('output_R_low',tf.expand_dims(output_R_low[1,:,:,:],0))  
  tf.summary.image('output_R_high',tf.expand_dims(output_R_high[1,:,:,:],0))
  tf.summary.image('output_C_low',tf.expand_dims(output_C_low[1,:,:,:],0))  
  tf.summary.image('output_C_high',tf.expand_dims(output_C_high[1,:,:,:],0))          
  tf.summary.image('output_I_low',tf.expand_dims(output_I_low[1,:,:,:],0))
  tf.summary.image('output_I_high',tf.expand_dims(output_I_high[1,:,:,:],0))
summary_op = tf.summary.merge_all()
train_writer = tf.summary.FileWriter('./log' + '/IID_train',sess.graph,flush_secs=60)
#load data
###train_data
train_low_data = []
train_high_data = []
train_low_data_names = glob(args.train_data_dir +'/low/*') 
train_low_data_names.sort()
train_high_data_names = glob(args.train_data_dir +'/high/*') 
train_high_data_names.sort()
assert len(train_low_data_names) == len(train_high_data_names)
print('[*] Number of training data: %d' % len(train_low_data_names))
for idx in range(len(train_low_data_names)):
    low_im = load_images_no_norm(train_low_data_names[idx])
    train_low_data.append(low_im)
    high_im = load_images_no_norm(train_high_data_names[idx])
    train_high_data.append(high_im)



epoch = 4000
learning_rate = 0.0001

train_phase = 'decomposition'
numBatch = len(train_low_data) // int(batch_size)
train_op = train_op_Decom
train_loss = loss_Decom
saver = saver_Decom

checkpoint_dir = './checkpoint/IID/'
if not os.path.isdir(checkpoint_dir):
    os.makedirs(checkpoint_dir)
ckpt=tf.train.get_checkpoint_state(checkpoint_dir)
if ckpt:
    print('loaded '+ckpt.model_checkpoint_path)
    saver.restore(sess,ckpt.model_checkpoint_path)
else:
    print('No decomnet pretrained model!')

start_step = 0
start_epoch = 0
iter_num = 0

print("[*] Start training for phase %s, with start epoch %d start iter %d : " % (train_phase, start_epoch, iter_num))
start_time = time.time()
image_id = 0
counter = 0
for epoch in range(start_epoch, epoch):
    for batch_id in range(start_step, numBatch):
        batch_input_low = np.zeros((batch_size, patch_size, patch_size, 3), dtype="float32")
        batch_input_high = np.zeros((batch_size, patch_size, patch_size, 3), dtype="float32")
        for patch_id in range(batch_size):
            h, w, _ = train_low_data[image_id].shape
            x = random.randint(0, h - patch_size)
            y = random.randint(0, w - patch_size)
            rand_mode = random.randint(0, 7)
            batch_input_low[patch_id, :, :, :] = data_augmentation(train_low_data[image_id][x : x+patch_size, y : y+patch_size, :], rand_mode)
            batch_input_high[patch_id, :, :, :] = data_augmentation(train_high_data[image_id][x : x+patch_size, y : y+patch_size, :], rand_mode)
            image_id = (image_id + 1) % len(train_low_data)
            if image_id == 0:
                tmp = list(zip(train_low_data, train_high_data))
                random.shuffle(tmp)
                train_low_data, train_high_data  = zip(*tmp)
        counter += 1
        _, loss,summary_str = sess.run([train_op, train_loss,summary_op], feed_dict={input_low: batch_input_low, \
                                                              input_high: batch_input_high, \
                                                              lr: learning_rate})
        print("%s Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.6f" \
              % (train_phase, epoch + 1, batch_id + 1, numBatch, time.time() - start_time, loss))
        train_writer.add_summary(summary_str,counter)
        iter_num += 1

    if epoch % 2 ==0:     
      saver.save(sess, checkpoint_dir + 'model.ckpt',global_step=epoch+1)

print("[*] Finish training for phase %s." % train_phase)
