import torch
import numpy as np
from os import listdir
import cv2
import os
from scipy import ndimage
from torchvision import transforms
from imageio import imread

def list_images(directory):
    images = []
    names = []
    dir = listdir(directory)
    dir.sort()
    for file in dir:
        name = file.lower()
        if name.endswith('.png'):
            images.append( file)
        elif name.endswith('.jpg'):
            images.append(file)
        elif name.endswith('.jpeg'):
            images.append(file)
        name1 = name.split('.')
        names.append(name1[0])
    return images

def get_img(path, mode='L'):
    """Load an image and return both the image array and associated filename"""
    if mode == 'L':
        image = imread(path, pilmode="L")
    
    # Extract filename without extension for mask lookup
    filename = os.path.basename(path).split('.')[0]
    
    return image, filename

def get_train_images_auto(paths, height=256, width=256, mode='RGB'):
    if isinstance(paths, str):
        paths = [paths]
    images = []
    filenames = []
    
    for path in paths:
        image, filename = get_img(path, mode=mode)
        
        # Resize if needed
        if height is not None and width is not None:
            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
            
        if mode == 'L':
            image = np.reshape(image, [1, image.shape[0], image.shape[1]])
        else:
            image = np.reshape(image, [image.shape[2], image.shape[0], image.shape[1]])
        
        images.append(image)
        filenames.append(filename)

    images = np.stack(images, axis=0)
    images = torch.from_numpy(images).float()
    
    # Attach filenames as attribute to tensor for later use with precomputed masks
    images.filenames = filenames
    
    return images

def get_test_images(paths, height=None, width=None, mode='L'):
    ImageToTensor = transforms.Compose([transforms.ToTensor()])
    if isinstance(paths, str):
        paths = [paths]
    images = []
    filenames = []
    
    for path in paths:
        image, filename = get_img(path, mode=mode) 
        
        # Resize if needed
        if height is not None and width is not None:
            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
            
        w, h = image.shape[0], image.shape[1]
        w_s = 256 - w % 256
        h_s = 256 - h % 256
        image = cv2.copyMakeBorder(image, 0, w_s, 0, h_s, cv2.BORDER_CONSTANT,
                                     value=128)
        if mode == 'L':
            image = np.reshape(image, [1, image.shape[0], image.shape[1]])
        else:
            image = ImageToTensor(image).float().numpy()*255
            
        images.append(image)
        filenames.append(filename)
        
    images = np.stack(images, axis=0)
    images = torch.from_numpy(images).float()
    
    # Attach filenames as attribute to tensor for later use with precomputed masks
    images.filenames = filenames
    
    return images

def save_images(path, data, out):
    w, h = out.shape[0], out.shape[1]
    if data.shape[1] == 1:
        data = data.reshape([data.shape[2], data.shape[3]])
    ori = data[0:w, 0:h]
    cv2.imwrite(path, ori)
