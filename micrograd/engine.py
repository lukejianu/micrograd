class Value:
  def __init__(self, data, children=(), op=''):
    self.data = data
    self.grad = 0
    self._children = children
    self._op = op

  def __add__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    return Value(self.data + other.data, (self, other), '+')

  def __mul__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    return Value(self.data * other.data, (self, other), '*')

  def __pow__(self, other):
    assert isinstance(other, (int, float)), "Exponent must be an int or float."
    return Value(self.data ** other, (self, other), '**')

  def relu(self):
    do_relu = lambda x: 0 if x < 0 else x
    return Value(do_relu(self.data), (self,), 'ReLU') 

  def __neg__(self):
    return self * -1

  def __radd__(self, other):
    return self + other

  def __sub__(self, other):
    return self + (-other)

  def __rsub__(self, other):
    return other + (-self)

  def __rmul__(self, other):
    return self * other

  def __truediv__(self, other):
    return self * other**-1

  def __rtruediv__(self, other):
    return other * self**-1

  def backward(self):
    topo = []
    visited = set()

    def build_topo(v):
      if v not in visited:
        visited.add(v)
        for child in v._children:
          if isinstance(child, Value): build_topo(child) 
        topo.append(v)

    build_topo(self)

    self.grad = 1
    for node in reversed(topo):
      node._backward()

  def _backward(self):
    do_dx = self.grad
    if (self._op == '+'):
      l, r = self._children
      dx_dl = dx_dr = 1
      l.grad += dx_dl * do_dx
      r.grad += dx_dr * do_dx
    elif (self._op == '*'):
      l, r = self._children
      dx_dl = r.data
      dx_dr = l.data
      l.grad += dx_dl * do_dx
      r.grad += dx_dr * do_dx
    elif (self._op == '**'):
      b, exp = self._children
      dx_db = exp * b.data ** (exp - 1)
      b.grad += dx_db * do_dx
    elif (self._op == 'ReLU'):
      (c,) = self._children
      dx_dc = 1 if c.data > 0 else 0
      c.grad += dx_dc * do_dx

  def __repr__(self):
    return f"Value(data={self.data}, grad={self.grad})"
