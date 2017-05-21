import numpy as np
from chainer import Chain, Function, Variable
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
