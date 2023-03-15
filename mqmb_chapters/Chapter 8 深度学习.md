# Transformer
人类的注意力：
- 持续式注意力
- 选择式注意力
- 交替式注意力
- 分配式注意力
深度学习中的注意力（Y.Bengio 2014年引入）：动态的将注意力注入到输入的序列中。
人类的记忆力：将信息进行编码、存储和记忆信息和过去的经历
- 感官记忆(sensory memory): 瞬间对环境的记忆
- 短期记忆(short-term memory)：
- 长期记忆(long-term memory):
感官记忆 -> 短期记忆：基于注意力机制提取感官记忆中关键信息储存
```ad-note
LSTM 和 GRU 模型都不具备长期记忆的能力
```
循环网络的标准架构：
- one-to-many: 句子分类
- many-to-one：图片描述
- mang-to-many：机器翻译、语言模型（自回归式的预测）
![[Pasted image 20230306173207.png]]
语言建模是通过自回归预测得到下一时刻词表的条件概率分布来采样下一个时刻可能出现的词
![[Pasted image 20230306173939.png]]
Seq-to-seq的many-to-many 模型：将encode编码的最后一个时刻的隐状态输入到decode
$$
\begin{aligned}
& p\left(y_1, \ldots, y_{T^{\prime}} \mid x_1, \ldots, x_T\right) \\
& =\prod_{t=1}^{T^{\prime}} p\left(y_t \mid c, y_1, \ldots, y_{t-1}\right) \\
& =\prod_{t=1}^{T^{\prime}} p\left(y_t \mid c, y_1, \ldots, y_{t-1}\right) \\
& =\prod_{t=1}^{T^{\prime}} g\left(c, s_{t-2}, y_{t-1}\right)
\end{aligned}
$$
其中encoder建模的是$x_1, \ldots, x_T$, 而decoder建模的是$y_1, \ldots, y_{T}$, 而整个网络学习的就是基于$x_1, \ldots, x_T$ 生成$y_1, \ldots, y_{T}$的条件概率。为什么要算这个编码，因为这个概率分布太大了，只能通过函数逼近来对高维的条件概率进行建模。
缺点：
- 信息损失，info loss(将历史数据编码到一个隐藏状态上肯定会有信息损失)
- 梯度爆炸/消杀
![[Pasted image 20230306174428.png]]
循环神经网络服从马尔可夫假设只能看到上一时刻(隐式的看到了历史数据)，因此这个模型实际是一个短视的模型，Attention机制可以看到整个序列，从一个大的范围内来分配序列的重要性。
| 局部网络 | 全局网络    |
| -------- | ----------- |
| CNN      | MLP         |
| LSTM     | Transformer |
如何分配注意力->相关性$e_{i j}=a\left(s_{i-1}, x_j\right)$
- 神经网络：$e_{i j}=v_a^{\top} \tanh \left(W_a s_{i-1}+U_a h_j\right)$
- 内积: $e_{i j}=v_a^{\top}s_{i-1}\dot{x_{j}}$
- normalize: softmax - $\alpha_{i j}=\frac{\exp \left(e_{i j}\right)}{\sum_{k=1}^T \exp \left(e_{i k}\right)}$
![[Pasted image 20230306181452.png]]
