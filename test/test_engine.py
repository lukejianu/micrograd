from micrograd.engine import Value

def test_foo():
  a = Value(2)
  b = Value(-3)
  c = a + b
  d = Value(4)
  L = d * c

  # dL/dL = 1
  # dL/dd = -1
  # dL/dc = 4
  # dL/da = 4
  # dL/db = 4
  assert L._data == -4
  assert c._data == -1
  assert a._grad == 4
  assert b._grad == 4
  # TODO: Make some claims about L

