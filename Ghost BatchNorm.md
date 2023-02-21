本文旨在解决 "泛化差距 "的问题。似乎神经网络在对大批量数据进行训练时，对未见过的数据往往表现得更差。  
  
为解决这个问题而提出的方法之一是将 batchnorm 层从计算统计数据（记住 BatchNorm 层改变了输入数据，使其成为 "正态分布"，平均数为 0，标准差为 1，从使用整个批次，改为使用批次的小部分，我们可以称之为纳米批次。

![[Pasted image 20220815163334.png]]
-   Calculate mean of each nano batch.
-   Calculate std of each nano batch.
-   Update running mean using an exponential moving average.
-   Update running std using an exponential moving average.
-   Return whole batch normalized.

![[Pasted image 20220815163807.png]]

```python:GBN.py
class GBN(torch.nn.Module):
    """
    Ghost Batch Normalization
    https://arxiv.org/abs/1705.08741
    """

    def __init__(self, input_dim, 
			    virtual_batch_size=128, momentum=0.01):
        super(GBN, self).__init__()

        self.input_dim = input_dim
        self.virtual_batch_size = virtual_batch_size
        self.bn = BatchNorm1d(self.input_dim, momentum=momentum)

    def forward(self, x):
        chunks = x.chunk(int(np.ceil(
			        x.shape[0] / self.virtual_batch_size)), 0)
        res = [self.bn(x_) for x_ in chunks]
        return torch.cat(res, dim=0)
        ```
