# PytorchNeuralStyleTransfer

Code to run Neural Style Transfer from our paper [Image Style Transfer Using Convolutional Neural Networks](http://www.cv-foundation.org/openaccess/content_cvpr_2016/html/Gatys_Image_Style_Transfer_CVPR_2016_paper.html).

Also includes coarse-to-fine high-resolution from our paper [Controlling Perceptual Factors in Neural Style Transfer](https://arxiv.org/abs/1611.07865).

To run the code you need to get the pytorch VGG19-Model from [Simonyan and Zisserman, 2014](https://arxiv.org/abs/1409.1556) by running: 

`sh download_models.sh`

Examples:
  python ./NeuralStyleTransfer.py --pyramid-levels 2 --style-image ./Images/dali-brasil.jpg --image-size 720 --style-size 512 --cuda 2 --iterations 200 --lr 1.0 --model-name vgg16_4x --content-image ./Images/bayou-hd.jpg --optimizer adam --eps 1e-4 --beta1 0.9 --half

python ./NeuralStyleTransfer.py --pyramid-levels 2 --style-image ./Images/dali-brasil.jpg --image-size 720 --style-size 512 --cuda 2 --iterations 1000 --lr 1.0 --model-name vgg16_4x --content-image ./Images/bayou-hd.jpg --optimizer adam --eps 1e-4 --beta1 0.9 --half 

Have fun :-)
