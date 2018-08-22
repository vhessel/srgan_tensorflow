#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 01:44:39 2018

@author: hessel
"""

from multiprocessing import Process, Queue
import numpy as np
from PIL import Image

def _load_data(files, image_queue, shuffle, img_size):
    while True:
        if shuffle:
            files = files[np.random.permutation(files.shape[0])]
        
        for f in files:
            img = Image.open(f)
            if img.size[0]<img_size[1] or img.size[1]<img_size[0]:
                continue
            img = img.convert("RGB")
            x_crop = np.random.randint(0,img.size[0]-img_size[1]+1)
            y_crop = np.random.randint(0,img.size[1]-img_size[0]+1)
            img = np.array(img.crop(box=(x_crop,y_crop,x_crop+img_size[1],y_crop+img_size[0])), dtype=np.uint8)
            image_queue.put(img)            

def BatchGenerator(files, batch_size, cache_size = 5, image_size=(96,96)):
    image_queue = Queue(maxsize=max(1,batch_size*cache_size))
    p_load = Process(target=_load_data, args=(files,image_queue,True,image_size))
    p_load.start()
    
    batch_data = np.zeros((batch_size,)+image_size+(3,))
    while True:
        for i in range(batch_size):
            batch_data[i] = image_queue.get()                
             
        yield batch_data.copy()