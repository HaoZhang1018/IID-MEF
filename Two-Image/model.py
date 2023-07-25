import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers.python.layers import initializers
from utils import *

def lrelu(x, trainbable=None):
    return tf.maximum(x*0.2,x)

def upsample_and_concat(x1, x2, output_channels, in_channels, scope_name, trainable=True):
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
        pool_size = 2
        deconv_filter = tf.get_variable('weights', [pool_size, pool_size, output_channels, in_channels], trainable= True)
        deconv = tf.nn.conv2d_transpose(x1, deconv_filter, tf.shape(x2) , strides=[1, pool_size, pool_size, 1], name=scope_name)

        deconv_output =  tf.concat([deconv, x2],3)
        deconv_output.set_shape([None, None, None, output_channels*2])

        return deconv_output



def DecomNet(input,training=True):
    with tf.variable_scope('DecomNet', reuse=tf.AUTO_REUSE):
        conv1=slim.conv2d(input,32,[3,3], rate=1, activation_fn=lrelu,scope='De_conv1')
        pool1=slim.max_pool2d(conv1, [2, 2], stride = 2, padding='SAME' )
        conv2=slim.conv2d(pool1,64,[3,3], rate=1, activation_fn=lrelu,scope='De_conv2')
        pool2=slim.max_pool2d(conv2, [2, 2], stride = 2, padding='SAME' )
        conv3=slim.conv2d(pool2,128,[3,3], rate=1, activation_fn=lrelu,scope='De_conv3')
        up1 =  upsample_and_concat( conv3, conv2, 64, 128 , 'up_1')
        conv4=slim.conv2d(up1,  64,[3,3], rate=1, activation_fn=lrelu,scope='De_conv4')
        up2 =  upsample_and_concat( conv4, conv1, 32, 64 , 'up_2')
        conv5=slim.conv2d(up2,  32,[3,3], rate=1, activation_fn=lrelu,scope='De_conv5')
        
        R_conv6=slim.conv2d(conv5,  16,[3,3], rate=1, activation_fn=lrelu,scope='R_De_conv6')
        R_conv7=slim.conv2d(R_conv6,1,[1,1], rate=1, activation_fn=None, scope='R_De_conv7')
        R_out = tf.sigmoid(R_conv7)   #### Reflectance Structure
        

        C_conv6=slim.conv2d(conv5,  16,[3,3], rate=1, activation_fn=lrelu,scope='C_De_conv6')
        C_conv7=slim.conv2d(C_conv6,3,[1,1], rate=1, activation_fn=None, scope='C_De_conv7')
        C_out = tf.sigmoid(C_conv7)   #### Reflectance Colors

        l_conv2=slim.conv2d(conv1,32,[3,3], rate=1, activation_fn=lrelu,scope='l_conv1_2')
        l_conv3=tf.concat([l_conv2, conv5],3)
        l_conv4=slim.conv2d(l_conv3,1,[1,1], rate=1, activation_fn=None,scope='l_conv1_4')
        L_out = tf.nn.softplus(l_conv4)     #### Illumination 

    return R_out, C_out, L_out


def RFusNet(R_1, R_2,training=True):
    with tf.variable_scope('RFusNet', reuse=tf.AUTO_REUSE):
        cat_input=tf.concat([R_1,R_2],axis=-1)
    
        conv1=slim.conv2d(cat_input,16,[3,3], rate=1, activation_fn=lrelu,scope='RF_conv1')
        conv2=slim.conv2d(conv1,16,[3,3], rate=1, activation_fn=lrelu,scope='RF_conv2')
        conv3=slim.conv2d(conv2,16,[3,3], rate=1, activation_fn=lrelu,scope='RF_conv3')

        cat_23=tf.concat([conv2,conv3],axis=-1)
        conv4=slim.conv2d(cat_23,16,[3,3], rate=1, activation_fn=lrelu,scope='RF_conv4')

        cat_14=tf.concat([conv1,conv4],axis=-1)
        conv5=slim.conv2d(cat_14,16,[3,3], rate=1, activation_fn=lrelu,scope='RF_conv5')
        
        conv6=slim.conv2d(conv5,4,[3,3], rate=1, activation_fn=lrelu,scope='RF_conv6')
        conv7=slim.conv2d(conv6,1,[3,3], rate=1, activation_fn=None,scope='RF_conv7')
        RFus_out =tf.sigmoid(conv7)
    return RFus_out



def SFusNet(S_1, S_2,training=True):
    with tf.variable_scope('SFusNet', reuse=tf.AUTO_REUSE):
        cat_input=tf.concat([S_1,S_2],axis=-1)
    
        conv1=slim.conv2d(cat_input,16,[5,5], rate=1, activation_fn=lrelu,scope='SF_conv1')
        
        conv2_3x3=slim.conv2d(conv1,16,[3,3], rate=1, activation_fn=lrelu,scope='SF_conv2_3x3')
        conv2_5x5=slim.conv2d(conv1,16,[5,5], rate=1, activation_fn=lrelu,scope='SF_conv2_5x5')
        conv2_7x7=slim.conv2d(conv1,16,[7,7], rate=1, activation_fn=lrelu,scope='SF_conv2_7x7')

        cat_conv2=tf.concat([conv2_3x3,conv2_5x5, conv2_7x7],axis=-1)
        
        conv3=slim.conv2d(cat_conv2,16,[5,5], rate=1, activation_fn=lrelu,scope='SF_conv3')
        
        conv4_3x3=slim.conv2d(conv3,16,[3,3], rate=1, activation_fn=lrelu,scope='SF_conv4_3x3')
        conv4_5x5=slim.conv2d(conv3,16,[5,5], rate=1, activation_fn=lrelu,scope='SF_conv4_5x5')
        conv4_7x7=slim.conv2d(conv3,16,[7,7], rate=1, activation_fn=lrelu,scope='SF_conv4_7x7')

        cat_conv4=tf.concat([conv4_3x3,conv4_5x5, conv4_7x7],axis=-1)
        
        conv5=slim.conv2d(cat_conv4,16,[5,5], rate=1, activation_fn=lrelu,scope='SF_conv5')
        
        conv6=slim.conv2d(conv5,4,[5,5], rate=1, activation_fn=lrelu,scope='SF_conv6')
        conv7=slim.conv2d(conv5,1,[5,5], rate=1, activation_fn=None,scope='SF_conv7')        
        SFus_out =tf.nn.softplus(conv7)
    return SFus_out



def CFusNet(C_1, C_2,training=True):
    with tf.variable_scope('CFusNet', reuse=tf.AUTO_REUSE):
        cat_input=tf.concat([C_1,C_2],axis=-1)
    
        conv1=slim.conv2d(cat_input,16,[3,3], rate=1, activation_fn=lrelu,scope='CF_conv1')
        
        CA1_max=tf.reduce_max(conv1, axis=(1, 2), keepdims=True)
        CA1_mean=tf.reduce_mean(conv1, axis=(1, 2), keepdims=True)
        CA1_max_conv1=slim.conv2d(CA1_max,4,[3,3], rate=1, activation_fn=lrelu,scope='CA1_max_conv1')
        CA1_mean_conv1=slim.conv2d(CA1_mean,4,[3,3], rate=1, activation_fn=lrelu,scope='CA1_mean_conv1')
        CA1_max_conv2=slim.conv2d(CA1_max,16,[3,3], rate=1, activation_fn=None,scope='CA1_max_conv2')
        CA1_mean_conv2=slim.conv2d(CA1_mean,16,[3,3], rate=1, activation_fn=None,scope='CA1_mean_conv2')
        CA1_map = tf.nn.sigmoid(CA1_max_conv2+CA1_mean_conv2) 
        conv1_out = conv1 * CA1_map
 
        conv2=slim.conv2d(conv1_out,16,[3,3], rate=1, activation_fn=lrelu,scope='CF_conv2')
        
        CA2_max=tf.reduce_max(conv2, axis=(1, 2), keepdims=True)
        CA2_mean=tf.reduce_mean(conv2, axis=(1, 2), keepdims=True)
        CA2_max_conv1=slim.conv2d(CA2_max,4,[3,3], rate=1, activation_fn=lrelu,scope='CA2_max_conv1')
        CA2_mean_conv1=slim.conv2d(CA2_mean,4,[3,3], rate=1, activation_fn=lrelu,scope='CA2_mean_conv1')
        CA2_max_conv2=slim.conv2d(CA2_max,16,[3,3], rate=1, activation_fn=None,scope='CA2_max_conv2')
        CA2_mean_conv2=slim.conv2d(CA2_mean,16,[3,3], rate=1, activation_fn=None,scope='CA2_mean_conv2')
        CA2_map = tf.nn.sigmoid(CA2_max_conv2+CA2_mean_conv2) 
        conv2_out = conv2 * CA2_map

        conv3=slim.conv2d(conv2_out,8,[3,3], rate=1, activation_fn=lrelu,scope='CF_conv3')
        conv4=slim.conv2d(conv3,3,[3,3], rate=1, activation_fn=None,scope='CF_conv4')
      
        CFus_out =tf.sigmoid(conv4)
    return CFus_out


