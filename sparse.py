import time
import torch

DEVICE = torch.device('cuda')

n = 30522
c = .01

t = torch.randn((n, n), device = DEVICE)
t[torch.rand_like(t) > c] = 0
t = t.to_sparse()
torch.cuda.empty_cache()

t1 = torch.sparse_coo_tensor(t._indices(), t._values(), size = (n,n))
t2 = torch.sparse_coo_tensor(t._indices().clone(), t._values().clone(), size = (n,n))

a =torch.randn((n,1), device = DEVICE)

def bb(t,n):
    t0 = time.monotonic()
    b = torch.mm(t, a).mul_(c)
    for i in range(n-2):
        b = torch.mm(t, b)
    b = torch.mm(t, b).mul_(c)
    torch.cuda.synchronize()
    print((time.monotonic()-t0)/n)

bb(t, 1000)
# 0.00034971597511321306

bb(t1, 1000)
# 0.03134377552382648

bb(t2, 1000)
# 0.030944300027564167


