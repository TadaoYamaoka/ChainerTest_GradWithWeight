import numpy as np
from chainer import Chain, Function, Variable
from chainer import optimizers
import chainer.functions as F
import chainer.links as L

W1 = [[[[-0.21877465,  0.81970602, -0.07861064],
        [-0.23030198, -0.58554208, -0.54126447],
        [ 0.18599892,  0.12918311, -0.11492223]]]]
W2 = [[-0.10862973, -0.24082854,  0.25668636, -0.19854707, -0.15225384,  0.62824327,  -0.45719072,  0.15064526, -0.24443959],
      [ 0.05131241,  0.24449363,  0.70825851, -0.25756082, -0.38595358, -0.15445025,  -0.15796955, -0.03341741,  0.01000506]]

img_size = 3
class MyChain(Chain):
    def __init__(self):
        super(MyChain, self).__init__(
            l1=L.Convolution2D(1, 1, ksize=3, pad=1, initialW=np.array(W1)),
            l2=L.Linear(img_size * img_size, 2, initialW=np.array(W2)),
        )

    def __call__(self, x):
        h = F.relu(self.l1(x))
        return self.l2(h)

model = MyChain()

optimizer = optimizers.SGD()
optimizer.setup(model)

# print param data
for path, param in model.namedparams():
    print(path)
    print(param.data)
print()

x_data = [[[[0.1, 0.4, 0.7],
            [0.3, 1.0, 0.5],
            [0.9, 0.5, 0.3]]],
          [[[1.0, 0.4, 0.6],
            [0.2, 0.1, 0.3],
            [0.5, 0.2, 0.9]]]]
t_data = [1, 0]

x = Variable(np.array(x_data, dtype=np.float32))
t = Variable(np.array(t_data, dtype=np.int32))

y = model(x)

model.cleargrads()
loss = F.softmax_cross_entropy(y, t)
loss.backward()

optimizer.update()

# print param data and grad
for path, param in model.namedparams():
    print(path)
    print(param.data)
    print(param.grad)
