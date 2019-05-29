import torch
import torchvision
from torchvision.utils import save_image
import torch.optim as optim
from utils import *

alpha = 0.035

style_weights = {'conv1_1':1.0,
                 'conv2_1':0.75,
                 'conv3_1':0.2,
                 'conv4_1':0.2,
                 'conv5_1':0.2}

content_weight = 1
style_weight = 1e6
save_img()
print("Loading model")
model = get_vgg()
print("Model Loaded")

content_features = get_features(content_image, model)
style_features = get_features(style_image, model)

style_grams = {layer:gram_matrix(style_features[layer]) for layer in style_features}

target = content_image.clone().requires_grad_(True).to(DEVICE)

show_every = 400
optimizer = optim.Adam([target], lr=alpha)
steps = 2000

print("Starting training")

for ii in range(1, steps+1):
    target_features = get_features(target, model)
    content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2']) ** 2)
    style_loss = 0
    for layer in style_weights:
        target_feature = target_features[layer]
        target_gram = gram_matrix(target_feature)
        _, d, h, w = target_feature.shape
        style_gram = style_grams[layer]
        layer_style_loss = style_weights[layer] + torch.mean((target_gram - style_gram)**2)
        style_loss += layer_style_loss/(d*w*h)
        
    total_loss = content_weight*content_loss + style_weight*style_loss
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    if ii%show_every==0:
        print("Total loss: ", total_loss.item())
        plt.imshow(im_convert(target))
        plt.savefig("target_image_step_"+str(ii)+".png")

print("Done!")
