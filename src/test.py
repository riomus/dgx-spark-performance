import time
import torch
print(torch.__version__)
d = 8192
x = torch.randn(size=(d, d)).to(torch.bfloat16).to("cuda")
y = torch.randn(size=(d, d)).to(torch.bfloat16).to("cuda")

def fun(x):
    for _ in range(50):
        x = x @ y.T
    return x

for _ in range(10):
    fun(x)
    torch.cuda.synchronize()

tic = time.time()
repetitions = 100
for _ in range(repetitions):
    fun(x)
    torch.cuda.synchronize()
toc = time.time()
s = (toc - tic)
msec = 1e3 * s
tf = (d**3)  * 2 * 50 * repetitions / (1024 **4)
print(f"{msec=:.3f}")
tflops = tf / s
print(f"{tflops=:.3f}")