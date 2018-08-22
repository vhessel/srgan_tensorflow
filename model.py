#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  5 22:35:12 2018

@author: dollar
"""

import os
import numpy as np
import tensorflow as tf
import glob

class SRGAN():
    def __init__(self, **kwargs):
        import keras.backend as K
        
        self._kwargs = kwargs
        
        #PARAMETERS: DEFAULT VALUES
        self._model_version = self._kwargs.get('model_version', 1)
        self._model_name = self._kwargs.get('model_name', 'unamed')
        self._mode = self._kwargs.get('mode','inference')
        self._learning_rate = self._kwargs.get('learning_rate', 1e-4)
        self._random_seed = self._kwargs.get('random_seed', 141) 
        self._device = self._kwargs.get('device', '/cpu:0') 
        
        with tf.device(self._device):
        
            tf.set_random_seed(self._random_seed)
            
            if self._mode == 'training':
                self._is_training = True
                self._hr_img_shape = self._kwargs.get('hr_img_shape')
            else:
                self._is_training = False
                self._hr_img_shape = None
            
            self._tb_log_dir = "./tensorboard"
            
            self._session = tf.InteractiveSession()
            K.set_session(self._session)
            
            #Placeholders
            self._ph_lr_img = tf.placeholder(tf.float32, shape=(None,None,None,3))
            self._ph_is_training = tf.placeholder(tf.bool, name="ph_is_training")
            if self._is_training == True:
                self._ph_hr_img = tf.placeholder(tf.float32, shape=(None,)+self._hr_img_shape)
                self._training_lr_img = tf.image.resize_bicubic(self._ph_hr_img, np.array(self._hr_img_shape)[:2]/4, name="hr_img_downsampled")
            
            #Models
            with tf.variable_scope("generator") as scope:
                self._generator_infer = self._get_generator(self._ph_lr_img, is_training=False)
                scope.reuse_variables()
                if self._is_training == True:
                    self._generator = self._get_generator(self._training_lr_img, is_training=True)
                else:
                    self._generator = self._generator_infer             
                
            if self._is_training == True:
                #Losses
                with tf.variable_scope("discriminator") as scope:
                    self._discriminator_fake = self._get_discriminator(self._generator, is_training=True)
                    scope.reuse_variables()
                    self._discriminator_real = self._get_discriminator(self._ph_hr_img, is_training=True)                
                    
                with tf.variable_scope("pretrained_vgg19"):
                    RGB_MEAN_PIXELS = np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3)).astype(np.float32)                
                    #Convert RGB Normalized [0,1] Images to VGG19 Input Format
                    gen_vgg19 = (self._generator+1.)/2*255. - RGB_MEAN_PIXELS
                    gen_vgg19 = tf.reverse(gen_vgg19, axis=[-1])                
                    img_vgg19 = (self._ph_hr_img+1.)/2*255. - RGB_MEAN_PIXELS
                    img_vgg19 = tf.reverse(img_vgg19, axis=[-1])
                    self._vgg19 = tf.keras.applications.VGG19(include_top = False, input_shape=self._hr_img_shape, weights="imagenet")        
                    self._vgg19 = tf.keras.models.Model(self._vgg19.input, self._vgg19.get_layer('block5_conv4').output)
                     
                self._loss_vgg19 = 0.006*tf.losses.mean_squared_error(self._vgg19(img_vgg19), self._vgg19(gen_vgg19))
                self._loss_mse = tf.losses.mean_squared_error(self._ph_hr_img, self._generator)            
                self._loss_generator = 1e-3*tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self._discriminator_fake), logits=self._discriminator_fake))
                
                self._d_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self._discriminator_real), logits=self._discriminator_real))
                self._d_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(self._discriminator_fake), logits=self._discriminator_fake))
                self._discriminator_loss = self._d_fake + self._d_real
            
                with tf.variable_scope("generator") as scope:
                    self._g_bn = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope="generator")
                    with tf.control_dependencies(self._g_bn):
                        g_var_list = [x for x in tf.trainable_variables() if 'generator' in x.name]
                        self._g_train_step = tf.train.AdamOptimizer(learning_rate=self._learning_rate, beta1=0.9).minimize(self._loss_vgg19+self._loss_generator, var_list = g_var_list)
                
                with tf.variable_scope("discriminator") as scope:
                    self._d_bn = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope="discriminator")
                    with tf.control_dependencies(self._d_bn):
                        d_var_list=[x for x in tf.trainable_variables() if 'discriminator' in x.name]
                        self._d_train_step = tf.train.AdamOptimizer(learning_rate=self._learning_rate, beta1=0.9).minimize(self._discriminator_loss, var_list = d_var_list)
            
                self._saver_discriminator = tf.train.Saver({x.name : x for x in tf.global_variables() if "discriminator" in x.name})
                
                self._add_tensorboar_train_summaries()
            
            self._saver_generator = tf.train.Saver({x.name : x for x in tf.global_variables() if "generator" in x.name})
            
            var_list = [x for x in tf.global_variables() if "pretrained_vgg19" not in x.name]
            self._session.run(tf.variables_initializer(var_list=var_list))

        
    def save_models(self, save_discriminator=True):
        self._saver_generator.save(self._session, "./checkpoint/" + self._model_name + "/generator.ckpt")
        if save_discriminator:
            self._saver_discriminator.save(self._session, "./checkpoint/" + self._model_name + "/discriminator.ckpt")
        
    def restore_models(self, load_discriminator=False):
        files = glob.glob("./checkpoint/" + self._model_name + "/*")
        if np.any(["generator" in x for x in files]):
            self._saver_generator.restore(self._session, "./checkpoint/" + self._model_name + "/generator.ckpt")
        if load_discriminator:
            if np.any(["discriminator" in x for x in files]):
                self._saver_discriminator.restore(self._session, "./checkpoint/" + self._model_name + "/discriminator.ckpt")

    def train_step_generator(self, imgs_hr, summary_step=None):
        if summary_step is not None:
            _, summary = self._session.run([self._g_train_step, self._summaries_merged], feed_dict={self._ph_hr_img: imgs_hr, self._ph_is_training: True})
            self._file_writer.add_summary(summary, summary_step)
        else:
            self._session.run([self._g_train_step], feed_dict={self._ph_hr_img: imgs_hr, self._ph_is_training: True})
        loss_mse, loss_vgg19, loss_gan = self._session.run([self._loss_mse, self._loss_vgg19, self._loss_generator], feed_dict={self._ph_hr_img: imgs_hr, self._ph_is_training: False})
        
        return loss_mse,loss_vgg19, loss_gan
    
    def train_step_discriminator(self, imgs_hr):
        self._session.run([self._d_train_step], feed_dict={self._ph_hr_img: imgs_hr, self._ph_is_training: True})
        d_loss, d_real, d_fake = self._session.run([self._discriminator_loss, self._d_real, self._d_fake], feed_dict={self._ph_hr_img: imgs_hr, self._ph_is_training: False})
        return d_loss, d_real, d_fake
        
    def increase_resolution(self, imgs_lr):
        return self._session.run(self._generator_infer, feed_dict = {self._ph_lr_img : imgs_lr, self._ph_is_training: False})
    
    def _add_tensorboar_train_summaries(self):
        with tf.name_scope("Low_Resolution"):
            tf.summary.image("image_lr", self._training_lr_img)
        with tf.name_scope("Generated"):
            tf.summary.image("generator", self._generator)
        with tf.name_scope("High_Resolution"):
            tf.summary.image("image_hr", self._ph_hr_img)
        with tf.name_scope("Generator_Losses"):
            tf.summary.scalar("MSE",self._loss_mse)
            tf.summary.scalar("VGG19",self._loss_vgg19)
            tf.summary.scalar("GAN",self._loss_generator)
        with tf.name_scope("Discriminator_Losses"):
            tf.summary.scalar("Total",self._discriminator_loss)
            tf.summary.scalar("Real",self._d_real)
            tf.summary.scalar("Fake",self._d_fake)
        self._summaries_merged = tf.summary.merge_all()
        self._file_writer = tf.summary.FileWriter(self._tb_log_dir,graph=self._session.graph)
    
    def _get_generator(self, ph_input, is_training=False):
        pre_res = tf.layers.conv2d(ph_input, 64, 9, padding="same", activation=None, name="g_conv_1")
        pre_res = tf.keras.layers.PReLU(shared_axes=[1, 2], name="prelu")(pre_res)
        
        
        def residual_blocks(input, filters, name="block"):
            res = tf.layers.conv2d(input, filters, 3, padding="same", activation=None, name="g_rconv_"+name)
            res = tf.nn.relu(res)
            res = tf.layers.batch_normalization(res, momentum=0.8, training=self._ph_is_training if is_training else False, name="g_rbn_"+name)
            res = tf.layers.conv2d(res, filters, 3, padding="same", activation=None, name="g_rconv2_"+name)
            return res + input
        
        res = pre_res
        for i in range(16):
            res = residual_blocks(res, 64, name=str(i))
        
        post_res = tf.layers.conv2d(res, 64, 3, padding="same", activation=None, name="g_conv_2")
        post_res = tf.layers.batch_normalization(post_res, momentum=0.8, training=self._ph_is_training if is_training else False, name="g_bn_1")
        post_res = post_res + pre_res
        
        post_res = tf.layers.conv2d_transpose(post_res, 256, 3, strides=2, padding="same", activation=None, name="g_deconv_1")
        post_res = tf.layers.conv2d(post_res, 256, 3, padding="same", activation=tf.nn.relu, name="g_conv_3")
        
        post_res = tf.layers.conv2d_transpose(post_res, 256, 3, strides=2, padding="same", activation=None, name="g_deconv_2")
        post_res = tf.layers.conv2d(post_res, 256, 3, padding="same", activation=tf.nn.relu, name="g_conv_4")
        
        output = tf.layers.conv2d(post_res, 3, 3, padding="same", activation=tf.tanh, name="output")
        
        return output
    
    def _get_discriminator(self, ph_input, is_training=False):
        desc = tf.layers.conv2d(ph_input, 64, 3, padding="same", activation=tf.nn.relu, name="d_conv2d_1")
        
        def d_block(input, filters, strides, name="block"):
            block = tf.layers.conv2d(input, filters, 3, strides=strides, padding="same", activation=None, name="d_conv2d_"+name)
            block = tf.nn.leaky_relu(block, name="d_lrelu_"+name)
            block = tf.layers.batch_normalization(block, training=self._ph_is_training, momentum=0.8, name="d_bn_"+name)
            return block
        
        desc = d_block(desc, 64, 2, name="2")
        desc = d_block(desc, 128, 1, name="3")
        desc = d_block(desc, 128, 2, name="4")
        desc = d_block(desc, 256, 1, name="5")
        desc = d_block(desc, 256, 2, name="6")
        desc = d_block(desc, 512, 1, name="7")
        desc = d_block(desc, 512, 2, name="8")
        

        fully = tf.reshape(desc, [-1,self._hr_img_shape[0]*self._hr_img_shape[1]//16//16*512])
        fully = tf.layers.dense(fully, 1024, activation=None, name="d_dense_1")
        fully = tf.nn.leaky_relu(fully, name="d_lrelu_9")
        
        output = tf.layers.dense(fully, 1, activation=None, name="d_dense_logit")
        
        return output
    
    def export_generator(self):
        export_path_base = "./export/"
        export_path = os.path.join(
            tf.compat.as_bytes(export_path_base),
            tf.compat.as_bytes(str(self._model_version)))
        print('Exporting trained model to', export_path)
        builder = tf.saved_model.builder.SavedModelBuilder(export_path)
        
        # Creates the TensorInfo protobuf objects that encapsulates the input/output tensors
        tensor_info_input = tf.saved_model.utils.build_tensor_info(self._ph_lr_img)
        # output tensor info
        tensor_info_output = tf.saved_model.utils.build_tensor_info(self._generator_infer)
            
        prediction_signature = (
            tf.saved_model.signature_def_utils.build_signature_def(
            inputs={'lr_images': tensor_info_input},
            outputs={'hr_image': tensor_info_output},
            method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))
    
        builder.add_meta_graph_and_variables(
            self._session, [tf.saved_model.tag_constants.SERVING],
            signature_def_map={'predict_images':prediction_signature,
        })

        # export the model
        builder.save(as_text=True)
        print('Done exporting!')
