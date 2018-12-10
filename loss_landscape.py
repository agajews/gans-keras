import os
import sys
import keras
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from utils.utils import import_mnist
from models.dcgan import DCGAN
from models.wgan import WGAN
from models.cgan import CGAN
from models.infogan import InfoGAN
from keras import backend as K


def get_session():
    """ Limit session memory usage
    """
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

def parse_args(args):
    """ Parse arguments from command line input
    """
    parser = argparse.ArgumentParser(description='Training and testing scripts for various types of GAN Architectures')
    parser.add_argument('--type', type=str, default='DCGAN',  help='Choose from {DCGAN, WGAN, CGAN, InfoGAN}')
    parser.add_argument('--nb_epochs', type=int, default=100, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=128, help="Batch size")
    parser.add_argument('--visualize', type=bool, default=True, help="Results visualization")
    parser.add_argument('--model', type=str, help="Pre-trained weights path")
    parser.add_argument('--save_path', type=str, default='weights/',help="Pre-trained weights path")
    parser.add_argument('--gpu', type=int, help='GPU ID')
    parser.add_argument('--train', dest='train', action='store_true', help="Retrain model (default)")
    parser.add_argument('--no-train', dest='train', action='store_false', help="Test model")
    parser.set_defaults(train=True)
    return parser.parse_args(args)

def main(args=None):
    # Parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    # Check if a GPU Id was set
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    keras.backend.tensorflow_backend.set_session(get_session())

    # Load appropriate model:
    if args.type=='DCGAN': # Deep Convolutional GAN
        model = DCGAN(args)
    elif(args.type=='WGAN'): # Wasserstein GAN
        model = WGAN(args)
    elif(args.type=='CGAN'): # Conditional GAN
        model = CGAN(args)
    elif(args.type=='InfoGAN'): # InfoGAN
        model = InfoGAN(args)

    # Load pre-trained weights
    if args.model:
        model.load_weights(args.model)
    elif not args.train:
        raise Exception('Please specify path to pretrained model')

    # # Load MNIST Data, pre-train D for a couple of iterations and train model
    # if args.train:
    #     X_train, y_train, _, _, N = import_mnist()
    #     model.pre_train(X_train, y_train)
    #     model.train(X_train,
    #         bs=args.batch_size,
    #         nb_epoch=args.nb_epochs,
    #         nb_iter=2,
    #         y_train=y_train,
    #         save_path=args.save_path)

    # (Optional) Visualize results
    # if args.visualize:
    #     model.visualize()

    # X_train, y_train, _, _, N = import_mnist()
    layers = [0, 4, 7, 9, 11]
    old_params = []
    A = []
    B = []

    n_x = 50
    n_y = 50
    n_samples = 500

    for l in layers:
        W, b = model.G.layers[l].get_weights()
        old_params.append((W, b))

        A_W = np.random.randn(*W.shape)
        A_W /= np.linalg.norm(A_W.reshape(-1, A_W.shape[-1]), axis=0)
        A_W *= np.linalg.norm(W.reshape(-1, W.shape[-1]), axis=0)
        A.append(A_W)

        B_W = np.random.randn(*W.shape)
        B_W /= np.linalg.norm(B_W.reshape(-1, B_W.shape[-1]), axis=0)
        B_W *= np.linalg.norm(W.reshape(-1, W.shape[-1]), axis=0)
        B.append(B_W)

    xs = np.linspace(-3, 3, n_x)
    ys = np.linspace(-3, 3, n_y)
    loss = np.zeros((n_x, n_y))

    for i, x in enumerate(ys):
        for j, y in enumerate(ys):
            for A_W, B_W, (W, b), l in zip(A, B, old_params, layers):
                model.G.layers[l].set_weights((W + x * A_W + y * B_W, b))
            print((i, j))
            loss[i, j] = model.eval_gen_loss(n_samples)

    print('Done!')

    np.save('{}_loss_n_samples_{}_xlarge'.format(args.type, n_samples), loss)
    xx, yy = np.meshgrid(xs, ys)
    plt.contour(xx, yy, loss)
    # plt.show()
    plt.savefig('figures/{}_landscape_n_samples_{}_xlarge.png'.format(args.type, n_samples))


if __name__ == '__main__':
    main()
