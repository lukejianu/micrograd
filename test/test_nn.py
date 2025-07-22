from micrograd.engine import Value
from micrograd.nn import MLP 

def test_nn():
  nn = MLP(2, [4, 4, 1])
  assert len(nn.parameters()) == (2 * 4) + (4 * 4) + (4 * 1) + 11 - 2
  assert isinstance(nn([2, 5]), Value)
