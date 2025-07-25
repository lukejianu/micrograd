import torch

from micrograd.engine import Value

def test_karpathy_video_example():
  a = Value(2)
  b = Value(-3)
  c = Value(10)
  e = a * b
  d = e + c
  f = Value(-2)
  L = d * f

  assert L.data == -8
  L.backward()
  assert L.grad == 1
  assert a.grad == 6
  assert b.grad == -4
  assert c.grad == -2
  assert e.grad == -2
  assert d.grad == -2
  assert f.grad == d.data
  assert L.grad == 1

def test_backward_simple_1():
  a = Value(2)
  b = Value(-3)
  L = a + b

  assert L.data == -1
  L.backward()
  assert L.grad == 1
  assert a.grad == 1
  assert b.grad == 1

def test_backward_simple_2():
  a = Value(2)
  b = Value(-3)
  L = a * b

  assert L.data == -6
  L.backward()
  assert L.grad == 1
  assert a.grad == -3
  assert b.grad == 2

def test_backward_simple_3():
  a = Value(2)
  L = a ** 3

  assert L.data == 8
  L.backward()
  assert L.grad == 1
  assert a.grad == 12

def test_backward_simple_4():
  b = Value(-3)
  L = b * b
  L.backward()
  assert L.data == 9
  assert b.grad == -6

def test_backward_simple_5():
  x = Value(-4.0)
  z = (2 * x) + 2 + x
  L = z + z * x
  L.backward()
  assert L.data == 30
  assert z.grad == -3
  assert x.grad == -19

def test_karpathy_test_1():
  x = Value(-4.0)
  z = 2 * x + 2 + x
  q = z.relu() + z * x
  h = (z * z).relu()
  y = h + q + q * x
  y.backward()
  xmg, ymg = x, y

  x = torch.Tensor([-4.0]).double()
  x.requires_grad = True
  z = 2 * x + 2 + x
  q = z.relu() + z * x
  h = (z * z).relu()
  y = h + q + q * x
  y.backward()
  xpt, ypt = x, y

  # forward pass went well
  assert ymg.data == ypt.data.item()
  # backward pass went well
  assert xmg.grad == xpt.grad.item()

def test_karpathy_test_2():
  a = Value(-4.0)
  b = Value(2.0)
  c = a + b
  d = a * b + b**3
  c += c + 1
  c += 1 + c + (-a)
  d += d * 2 + (b + a).relu()
  d += 3 * d + (b - a).relu()
  e = c - d
  f = e**2
  g = f / 2.0
  g += 10.0 / f
  g.backward()
  amg, bmg, gmg = a, b, g

  a = torch.Tensor([-4.0]).double()
  b = torch.Tensor([2.0]).double()
  a.requires_grad = True
  b.requires_grad = True
  c = a + b
  d = a * b + b**3
  c = c + c + 1
  c = c + 1 + c + (-a)
  d = d + d * 2 + (b + a).relu()
  d = d + 3 * d + (b - a).relu()
  e = c - d
  f = e**2
  g = f / 2.0
  g = g + 10.0 / f
  g.backward()
  apt, bpt, gpt = a, b, g

  tol = 1e-6
  # forward pass went well
  assert abs(gmg.data - gpt.data.item()) < tol
  # backward pass went well
  assert abs(amg.grad - apt.grad.item()) < tol
  assert abs(bmg.grad - bpt.grad.item()) < tol
