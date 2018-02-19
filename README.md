# GANs Implementations in Keras  

Keras implementation of:  
- [Deep Convolutional GAN](https://arxiv.org/abs/1511.06434)  
- [Wasserstein GAN](https://arxiv.org/abs/1701.07875)  
- [Conditional GAN](https://arxiv.org/abs/1411.1784)  
- [InfoGAN](https://arxiv.org/abs/1606.03657)  

The DCGAN code was inspired by Jeremy Howard's [course](http://course.fast.ai/)  

### Requirements:  

You will need [Keras 1.2.2](https://pypi.python.org/pypi/Keras/1.2.2) with a Tensorflow backend.  
To install dependencies, run `pip install -r requirements.txt`  
For command line parameters explanations:
```shell
python main.py -h
```   

## DCGAN  
[Deep Convolutional GANs](https://arxiv.org/abs/1511.06434) was one of the first modifications made to the original GAN architecture to avoid mode collapsing. Theses improvements include:  
- Replacing pooling with strided convolutions
- Using Batch-Normalization in both G and D
- Starting G with a single Fully-Connected layer, end D with a flattening layer. The rest should be Fully-Convolutional
- Using LeakyReLU activations in D, ReLU in G, with the exception of the last layer of G which should be tanh  

**TODO** -> Add picture  
```shell
python main.py --type DCGAN --no-train --model weights/DCGAN.h5 # Running pretrained model
python main.py --type DCGAN # Retraining
```

## WGAN  
Following up on the DCGAN architecture, the [Wasserstein GAN](https://arxiv.org/abs/1701.07875) aims at leveraging another distance metric between distribution to train G and D. More specifically, WGANs use the EM distance, which has the nice property of being continuous and differentiable for feed-forward networks. In practice, computing the EM distance is intractable, but we can approximate it by clipping the discriminator weights. The insures that D learns a K-Lipschitz function to compute the EM distance. Additionally, we:  
- Remove the sigmoid activation from D, leaving no constraint to its output range
- Use RMSprop optimizer over Adam  

**TODO** -> Add picture  
```shell
python main.py --type WGAN --no-train --model weights/WGAN.h5 # Running pretrained model
python main.py --type WGAN # Retraining
```  

## cGAN  
[Conditional GANs](https://arxiv.org/abs/1411.1784) are a variant to classic GANs, that allow one to condition both G and D on an auxiliary input y. We do so simply feeding y through an additional input layer to both G and D. In practice we have it go through an initial FC layer. This allows us to have two variables when generating new images:
- The random noise vector z
- The conditional label y  

**TODO** -> Add picture  
```shell
python main.py --type CGAN --no-train --model weights/CGAN.h5 # Running pretrained model
python main.py --type CGAN # Retraining
```  

## InfoGAN  
The motivation behind the [InfoGAN](https://arxiv.org/abs/1606.03657) architecture is to learn a smaller dimentional, disentangled representation of the images to be generated. To do so, we introduce a latent code c, that is concatenated with the noise vector z. When training, we then want to maximize the mutual information between the latent code c and the generated image G(z,c). In practice, we:
- Feed c in G through an additional input layer
- Create an auxiliary head Q that shares some of its weights with D, and train it to maximize the mutual information I(c, G(z,c))   

**TODO** -> Add picture + explanations  
```shell
python main.py --type InfoGAN --no-train --model weights/InfoGAN.h5 # Running pretrained model
python main.py --type InfoGAN # Retraining
```   
