import numpy as np
from chainer import cuda, Chain, Function, Variable
from chainer import optimizers
import chainer.functions as F
import chainer.links as L

W1 = [[ 1.21082544, -0.42751756],
      [ 1.35623264, -0.1971387 ],
      [-0.77883673,  0.28367677]]
W2 = [[ 0.08621028, -0.19540818,  0.78203094],
      [ 0.30133799,  1.3698988 , -0.01031571]]

class MyChain(Chain):
    def __init__(self):
        super(MyChain, self).__init__(
            l1=L.Linear(2, 3, initialW=np.array(W1)),
            l2=L.Linear(3, 2, initialW=np.array(W2)),
        )

    def __call__(self, x):
        h = F.relu(self.l1(x))
        return self.l2(h)

model = MyChain()
model.to_gpu()

optimizer = optimizers.SGD()
optimizer.setup(model)

# print param data
for path, param in model.namedparams():
    print(path)
    print(param.data)
print()

x_data = [[0.1, 0.4],
          [0.2, 0.5],
          [0.3, 0.6]]
t_data = [1, 0, 0]
z_data = [1.0, 0.5, 0.5]

# init grads
dic_grads = {}
for path, param in model.namedparams():
    dic_grads[path] = []

for x_elem, t_elem in zip(x_data, t_data):
    x = Variable(cuda.to_gpu(np.array([x_elem], dtype=np.float32)))
    t = Variable(cuda.to_gpu(np.array([t_elem], dtype=np.int32)))

    y = model(x)

    model.cleargrads()
    loss = F.softmax_cross_entropy(y, t)
    loss.backward()

    # save grad
    for path, param in model.namedparams():
        dic_grads[path].append(cuda.to_cpu(param.grad))

# manipulate grad
z = np.array(z_data, dtype=np.float32)
for path, param in model.namedparams():
    grads = np.array(dic_grads[path], dtype=np.float32)
    grad = np.tensordot(z, grads, axes=1)
    grad /= len(z_data)
    param.grad = cuda.to_gpu(grad)

optimizer.update()

# print param data and grad
for path, param in model.namedparams():
    print(path)
    print(cuda.to_cpu(param.data))
    print(cuda.to_cpu(param.grad))
