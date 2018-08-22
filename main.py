#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 23:12:37 2018

@author: hessel
"""

from model import SRGAN
import argparse
from batch_generator import BatchGenerator
import numpy as np
import glob
import os
from PIL import Image
from tqdm import tqdm

def train_srgan(args):    
    np.random.seed(args.random_seed)
    
    hr_size = tuple(args.hr_size) + (3,)
    files = np.array(glob.glob(args.images_dir))
    files  = files[np.random.permutation(files.shape[0])]
    if files.shape[0]==0:
        raise FileNotFoundError("Training images not found")

    if args.log_images > 0:
        images_idx = np.random.randint(0,files.shape[0],args.log_images)
        files = np.setdiff1d(files,files[images_idx])
        
        tracking_images_lr = []
        tracking_images_hr = []
        for i,f in enumerate(images_idx):
            os.mkdir("images/%d"%i)
            img = Image.open(files[f])
            img.save("images/%d/high_resolution.png"%i)
            tracking_images_hr.append(np.array(img)/127.5-1.)
            orig_size = img.size
            img = img.resize((img.size[0]//4,img.size[1]//4), Image.BICUBIC)
            img.resize(orig_size , Image.NONE).save("images/%d/low_resolution.png"%i)
            img.resize(orig_size , Image.BICUBIC).save("images/%d/bicubic.png"%i)
            tracking_images_lr.append(np.array(img)/127.5-1.)
    
    files = files[:args.max_images]
    
    batch_generator = BatchGenerator(files, args.batch_size, cache_size=10, image_size=hr_size[:2])
    from model import SRGAN
    srgan = SRGAN(hr_img_shape=hr_size, random_seed = args.random_seed)
    srgan.restore_models()
    
    for step in tqdm(range(args.training_steps), total=args.training_steps):
        batch_data = next(batch_generator)/255.*2.-1
        srgan.train_step_generator(batch_data, summary_step=step if step%args.steps_tensorboard==0 else None)
        srgan.train_step_discriminator(batch_data)
        #LOG IMAGES
        if args.log_images > 0 and step%args.steps_log==0:
            for i,f in enumerate(tracking_images_lr):
                hr_generated = srgan.increase_resolution(np.expand_dims(f,0))[0]
                hr_generated = ((hr_generated+1.)*127.5).astype(np.uint8)
                img = Image.fromarray(hr_generated)
                img.save("images/%d/generated_step_%d.png"%(i,step))
        if step%args.steps_checkpoint==0:
            srgan.save_models()
    srgan.save_models()
    if args.verbose > 0:
        print("Training done")
    
def srgan_inference(args):
    if args.verbose > 0:
        print("Inference Mode")
    
    srgan = SRGAN(**vars(args))
    srgan.restore_models()
    
    inference_files = glob.glob(args.inference_dir)
    for f in tqdm(inference_files):
        lr_img = np.expand_dims(np.array(Image.open(f)),0)/127.5-1.
        hr_generated = srgan.increase_resolution(lr_img)[0]
        hr_generated = ((hr_generated+1.)*127.5).astype(np.uint8)
        Image.fromarray(hr_generated).save(os.path.join(args.output_dir, "hr_" + os.path.basename(f)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser("SRGAN Training")
    
    parser.add_argument('--verbose', action='store', type=int, default=0, dest='verbose', help="level of verbosity")
    parser.add_argument('--device', action='store', default='/cpu:0', dest='device', help='device used by tensorflow (usually: /gpu:0 or /cpu:0)')
    
    #Inference Parameters
    parser.add_argument('--inference-dir', action='store', default="./inference/*", dest='inference_dir', help="path of inference images (glob format. example: ~/data/training/**/*.jpg )")
    parser.add_argument('--output-dir', action='store', default="output", dest='output_dir', help="path to write super-resolution images")
    
    #Training Parameters
    parser.add_argument('--images-dir', action='store', dest='images_dir', help="path of training images (glob format. example: ~/data/training/**/*.jpg )")
    parser.add_argument('--batch-size', action='store', type=int, default=8, dest='batch_size', help="training batch size")
    parser.add_argument('--random-seed', action='store', default=142, dest='random_seed', help="")
    parser.add_argument('--training-steps', action='store', type=int,  default=100000, dest='training_steps', help="number of steps to run the training")
    parser.add_argument('--log-images', action='store', default=10, dest='log_images', help="number of images to log during training")
    parser.add_argument('--steps-log', action='store', type=int, default=500, dest='steps_log', help="number of training steps to log images")
    parser.add_argument('--steps-tensorboard', action='store', type=int, default=100, dest='steps_tensorboard', help="number of training steps to log tensorboard")
    parser.add_argument('--steps-checkpoint', action='store', type=int, default=1000, dest='steps_checkpoint', help="number of training steps to save a model checkpoint")
    parser.add_argument('--max-images', action='store', type=int, default=350000, dest='max_images', help="max number of images to use")
    
    #Model Parameters
    parser.add_argument('--model-version', action='store', type=int, default=1, dest='model_version', help="model version used in the tensorboard serving export")
    parser.add_argument('--model-name', action='store', default='celeba', dest='model_name', help="name used to save/restore the models")
    parser.add_argument('--mode', action='store', choices=['inference','training'], default='inference', dest='mode', help="mode to run the model: training/inference")
    parser.add_argument('--learning-rate', action='store', type=float, default=1e-5, dest='learning_rate', help="learning rate to use in the training algorithm")
    parser.add_argument('--hr-size', action='store', type=int, default=(320,320), dest='hr_size', nargs=2, help="high-resolution image size - format: height width")
    
    
    args = parser.parse_args()
    if args.mode == 'inference':
        srgan_inference(args)
    elif args.mode == 'training':
        train_srgan(args)
        



