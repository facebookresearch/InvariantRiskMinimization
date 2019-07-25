# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
import torch
from torchvision import datasets
from torch import nn, optim, autograd

# Load MNIST, make train/val splits, and shuffle train set examples

mnist = datasets.MNIST('/tmp', train=True, download=True)
mnist_train = (mnist.data[:50000], mnist.targets[:50000])
mnist_val = (mnist.data[50000:], mnist.targets[50000:])

rng_state = np.random.get_state()
np.random.shuffle(mnist_train[0].numpy())
np.random.set_state(rng_state)
np.random.shuffle(mnist_train[1].numpy())

# Build environments

def make_environment(images, labels, e):
  def torch_bernoulli(p, size):
    return (torch.rand(size) < p).float()
  def torch_xor(a, b):
    return (a-b).abs() # Assumes both inputs are either 0 or 1
  # 2x subsample for computational convenience
  images = images.reshape((-1, 28, 28))[:, ::2, ::2]
  # Assign a binary label y based on the digit; flip label with probability 0.25
  labels = (labels < 5).float()
  labels = torch_xor(labels, torch_bernoulli(0.25, len(labels)))
  # Assign a color based on the label; flip the color with probability e
  colors = torch_xor(labels, torch_bernoulli(e, len(labels)))
  # Apply the color to the image by zeroing out the other color channel
  images = torch.stack([images, images], dim=1)
  images[torch.tensor(range(len(images))), (1-colors).long(), :, :] *= 0
  return {
    'images': (images.float() / 255.).cuda(),
    'labels': labels[:, None].cuda()
  }

envs = [
  make_environment(mnist_train[0][::2], mnist_train[1][::2], 0.2),
  make_environment(mnist_train[0][1::2], mnist_train[1][1::2], 0.1),
  make_environment(mnist_val[0], mnist_val[1], 0.9)
]

# Define and instatiate the model

class MLP(nn.Module):
  def __init__(self):
    super(MLP, self).__init__()
    lin1 = nn.Linear(2 * 14 * 14, 256)
    lin2 = nn.Linear(256, 256)
    lin3 = nn.Linear(256, 1)
    self._main = nn.Sequential(lin1, nn.ReLU(True), lin2, nn.ReLU(True), lin3)
  def forward(self, input):
    out = input.view(input.shape[0], 2 * 14 * 14)
    out = self._main(out)
    return out
mlp = MLP().cuda()

# Define loss function helpers

def mean_nll(logits, y):
  return nn.functional.binary_cross_entropy_with_logits(logits, y)

def mean_accuracy(logits, y):
  preds = (logits > 0.).float()
  return ((preds - y).abs() < 1e-2).float().mean()

def penalty(logits, y):
  scale = torch.tensor(1.).cuda().requires_grad_()
  loss = mean_nll(logits, y * scale)
  grad = autograd.grad(loss, [scale], create_graph=True)[0]
  return torch.sum(grad**2)

# Train loop

optimizer = optim.Adam(mlp.parameters())

def pretty_print(*values):
  col_width = 13
  def format_val(v):
    if not isinstance(v, str):
      v = np.array2string(v, precision=5, floatmode='fixed')
    return v.ljust(col_width)
  str_values = [format_val(v) for v in values]
  print("   ".join(str_values))

pretty_print('step', 'train nll', 'train acc', 'train penalty', 'test acc')

for step in range(1000):
  for env in envs:
    logits = mlp(env['images'])
    env['nll'] = mean_nll(logits, env['labels'])
    env['acc'] = mean_accuracy(logits, env['labels'])
    env['penalty'] = penalty(logits, env['labels'])

  train_nll = (envs[0]['nll'] + envs[1]['nll']) / 2.
  train_acc = (envs[0]['acc'] + envs[1]['acc']) / 2.
  train_penalty = (envs[0]['penalty'] + envs[1]['penalty']) / 2.

  if step < 100:
    loss = train_penalty + train_nll
  else:
    loss = train_penalty + (1e-4 * train_nll)

  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  
  test_acc = envs[2]['acc']
  if step % 100 == 0:
    pretty_print(
      np.int32(step),
      train_nll.detach().cpu().numpy(),
      train_acc.detach().cpu().numpy(),
      train_penalty.detach().cpu().numpy(),
      test_acc.detach().cpu().numpy()
    )
