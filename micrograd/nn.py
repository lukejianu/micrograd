import random

from micrograd.engine import Value

class Module:
  def zero_grad(self):
    for p in self.parameters():
      p.grad = 0

class Neuron(Module):
  def __init__(self, nin, nonlin=True):
    self.ws = [Value(random.uniform(-1, 1)) for _ in range(nin)]
    self.b = Value(0)
    self.nonlin = nonlin

  def __call__(self, xs):
    act = sum((wi * xi for wi, xi in zip(self.ws, xs)), self.b)
    return act.relu() if self.nonlin else act

  def parameters(self):
    return self.ws + [self.b]

class Layer(Module):
  def __init__(self, nin, nout, **kwargs):
    self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

  def __call__(self, xs):
    acts = [neuron(xs) for neuron in self.neurons]
    return acts[0] if len(acts) == 1 else acts

  def parameters(self):
    return [p for neuron in self.neurons for p in neuron.parameters()]

class MLP(Module):
  def __init__(self, nin, nouts):
    sz = [nin] + nouts
    self.layers = [Layer(sz[i], sz[i + 1], nonlin=(i != len(nouts) - 1)) for i in range(len(nouts))]

  def __call__(self, xs):
    for layer in self.layers:
      xs = layer(xs)
    return xs

  def parameters(self):
    return [p for layer in self.layers for p in layer.parameters()]
