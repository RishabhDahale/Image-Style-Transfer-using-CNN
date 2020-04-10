# Image-Style-Transfer-using-CNN
In this script, https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf is implemented.
utils.py contains all the necessary help functions, train.py contains the training code. The CNN network is implemented in pytorch. The content_image and style_image can be changed from utils.py

## Results
Some of the results which I got are shown below:

Style Image 
<img src="https://github.com/RishabhDahale/Image-Style-Transfer-using-CNN/blob/master/results/original_images/mosiac.jpg" width="100">
Content Image
<img src="https://github.com/RishabhDahale/Image-Style-Transfer-using-CNN/blob/master/results/original_images/janelle.png" width="100">

Final Image
<img src="https://github.com/RishabhDahale/Image-Style-Transfer-using-CNN/blob/master/results/sample%201.png" width="150">


Style Image 
<img src="https://github.com/RishabhDahale/Image-Style-Transfer-using-CNN/blob/master/results/original_images/sunflower.jpg" width="100">
Content Image
<img src="https://github.com/RishabhDahale/Image-Style-Transfer-using-CNN/blob/master/results/original_images/img.jpeg" width="100">

Final Image
<img src="https://github.com/RishabhDahale/Image-Style-Transfer-using-CNN/blob/master/results/tree%201.png" width="150">


In the second result we can see that even thought the colou scheme of the original scenery was very different from the style image, the network was able to successfully isolate and change it.
