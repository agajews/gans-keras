import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


loss = np.load('WGAN_loss_n_samples_500_xlarge_epoch_1.npy')
xs = np.linspace(-2, 2, loss.shape[0])
ys = np.linspace(-2, 2, loss.shape[1])
xx, yy = np.meshgrid(xs, ys)

ax = plt.axes(projection='3d')
# ax.contour3D(xx, yy, loss, 50, cmap='binary')
ax.plot_surface(xx, yy, loss, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')

plt.show()
