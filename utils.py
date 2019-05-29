from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from torchvision import transforms, models
import torch.nn as nn


def load_image(imgae_path, max_size=400, shape=None):
    image = Image.open(imgae_path).convert("RGB")
    if max(image.size) > max_size:
        size = max_size
    else:
        size = max(image.size)
    if shape is not None:
        size=shape
    in_transform = transforms.Compose([
                        transforms.Resize(size),
                        transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406),
                                             (0.229, 0.224, 0.225))
    ])
    image = in_transform(image)[:3, :, :].unsqueeze(0)
    
    return image

def im_convert(tensor):
    image = tensor.to('cpu').clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1, 2, 0)
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    image = image.clip(0, 1)
    return image
    
def get_features(image, model, layers=None):
    if layers==None:
        layers = {'0':'conv1_1',
                  '5':'conv2_1',
                  '10':'conv3_1',
                  '19':'conv4_1',
                  '21':'conv4_2',
                  '28':'conv5_1'}
    
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x).to(DEVICE)
        if name in layers:
            features[layers[name]] = x
    return features
    
def gram_matrix(x):
    _, d, h, w = x.size()
    x = x.view(d, h*w)
    gram = torch.mm(x, x.t())
    return gram

DEVICE = ('cuda' if torch.cuda.is_available() else 'cpu')

def get_vgg():
    vgg = models.vgg19(pretrained=True).features
    for param in vgg.parameters():
        param.requires_grad_(False)
    vgg = vgg.to(DEVICE)
    return vgg

content_image = load_image("images/riya2.jpg").to(DEVICE)
style_image = load_image('images/janelle.png').to(DEVICE)

def save_img(content=content_image, style=style_image, dest=""):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(21, 10))
    ax1.imshow(im_convert(style))
    ax2.imshow(im_convert(content))
    plt.savefig(dest+"content_and_style_image.png", bbox_inches='tight')