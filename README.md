# cs231notbook
some convolution and mathematic deduction in the progression of doing assignments
## 卷积层反向传播推导导
这个运算可以通过链式法则来推导。假设我们有一个卷积层的输入 x，滤波器 w，以及输出特征图的上游梯度 dout。我们需要计算滤波器的梯度 dw。下面是推导过程：

首先，我们知道卷积层的前向传播计算可以表示为：
`out[n, f, j, i] = sum(x[n, :, j * stride:j * stride + HH, i * stride:i * stride + WW] * w[f])`
其中，n 表示样本索引，f 表示滤波器索引，j, i 表示输出特征图的空间位置。

我们的目标是计算滤波器的梯度 dw。根据链式法则，我们需要计算损失函数关于滤波器的梯度。假设损失函数为 L，那么有：
`dL/dw = sum(dL/dout[n, f, j, i] * dout[n, f, j, i]/dw)`

对于第 f 个滤波器的梯度，我们可以将其表示为：
`dw[f] = sum(dL/dout[n, f, j, i] * dout[n, f, j, i]/dw)`

现在我们来计算 `dout[n, f, j, i]/dw`。可以观察到，在卷积操作中，滤波器 w[f] 与输入切片` x[n, :, j * stride:j * stride + HH, i * stride:i * stride + WW]` 是通过点乘操作进行的。因此，我们可以得到：
`dout[n, f, j, i]/dw = x[n, :, j * stride:j * stride + HH, i * stride:i * stride + WW]`

最后，我们将 `dL/dout[n, f, j, i]` 乘以 `x[n, :, j * stride:j * stride + HH, i * stride:i * stride + WW]` 得到 `dw[f]` 的相应部分，然后在所有样本和位置的梯度上求和，即可得到完整的滤波器梯度。

综上所述，`dw[f] += x_pad[n, :, j * stride:j * stride + HH, i * stride:i * stride + WW] * dout[n, f, j, i]` 表示了将上游梯度 dout 与输入 x_pad 相应部分进行乘积运算，并累加到滤波器梯度 dw[f] 上的操作。
## 最大池化反向传播推导
反向传播计算最大池化层的梯度需要使用池化层的缓存以及上游梯度作为输入。下面是求解最大池化层反向传播的推导过程：

假设我们有最大池化层的输入 x，池化层的输出 out，以及上游梯度 dout。我们的目标是计算输入的梯度 dx。

首先，我们将 x 中的每个元素存储在 out 对应位置的最大值的索引中。我们将这些索引存储在 max_idx 中，它具有与 out 相同的形状。

接下来，我们将 dout 的梯度传播到 x 中对应位置的最大值。具体操作如下：

```python
dx = np.zeros_like(x)
N, C, H, W = x.shape

for n in range(N):<br>
    for c in range(C):<br>
        for h in range(H):<br>
            for w in range(W):<br>
                # 获取当前位置的最大值的索引<br>
                `idx = max_idx[n, c, h, w]'<br>
                # 将上游梯度传递给最大值的位置<br>
                dx[n, c, h, w] = dout[n, c, h, w] * (idx == x[n, c, h, w])`<br>
上述代码中，我们遍历输入 x 的每个位置 (n, c, h, w)。我们使用 max_idx 中相应位置的索引来检索 x 中的最大值位置。然后，我们将上游梯度 dout 乘以一个条件 (idx == `x[n, c, h, w])`，这个条件是指只有在 x 中的元素与最大值相等时才传递梯度，否则为零。

最后，通过以上操作，我们可以计算出输入 x 的梯度 dx。
## relu激活函数
f(x) = max(0, x) \<br>

