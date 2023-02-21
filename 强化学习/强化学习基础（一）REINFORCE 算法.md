Author: Zehoo Pu
Created: May 5, 2022 4:12 PM
Last edit time: August 12, 2022 1:09 PM
Tags: #Machine_Learning #强化学习
### 1. Introduction
**Reinforce算法**，也就是平常所说的策略网络。算法通过学习一个参数化的网络来生成对应状态$S_t$下不同动作$a_t^i$的概率分布，即构建环境状态与动作之间的映射$f:s\rightarrow t$，因此属于**策略梯度算法**。

**算法关键要素：**

- 参数化的梯度（即训练好的神经网络）
- 优化目标（最大化期望回报）
- 更新策略参数的方法

通过梯度更新来优化策略网络最大化期望回报，在可选的行动中选择概率最大的行动

### 2. Policy

策略$\pi$是一个从环境状态映射到动作的函数，智能体可以学习策略并使用该策略在环境中采样动作，最优的策略$\pi^*$被认为是最大化累计折现回报的策略。因此，可以选择使用神经网络这样的参数化模型对这一函数进行近似，并设置合适的目标函数来更新参数。

对于不同的参数，$\pi_{\theta_1}$和$\pi_{\theta_2}$实际代表两个不同的策略，因为对于不同的参数在环境中采样动作的结果是不一样的，所以对于一个神经网络来说可以通过不同的参数代表不同的策略，而当参数更新变化时策略也随之变化。

### 3. The Objective Function

在强化学习中，一条轨迹(trajectory)可以表示为：$\tau = s_0,a_0,r_0,\cdots,s_T,a_T,r_T$，对于一条轨迹从t时刻开始的累计折旧回报$R_t(\tau)$可以被定义为：

$$
R_t(\tau) = \sum_{t^{'}=t}^T\gamma^{t^{'}-t}r^{'}_t
$$

<aside>
💡 从t时刻开始算起累计折旧回报误差，但折旧因子的幂要从0开始

</aside>

对于策略$\pi$而言，其目标函数是最大化整条轨迹的累计折旧期望回报，可以被定义为：

$$
J(\pi_\theta) = \mathbb{E}_{\tau \sim\pi_{\theta}}[\sum_{t=0}^{T}\gamma^tr_t]
$$

上式中的期望是算法通过策略$\pi$采样多条轨迹进行计算来近似期望，但其结果是随着策略的变化而变化，这一点决定了REINFORCE算法是一种on-policy的算法。

### 4. 策略梯度

对于一个策略$\pi_0$和目标函数$J(\pi_\theta)$，实际需要解决的优化问题是：

$$
\max_{\theta} J(\pi_\theta) = \mathbb{E}_{\tau\sim\pi_\theta}[R(\tau)]
$$

为了最大化目标函数，采用梯度下降算法来更新策略参数$\theta$:

$$
\theta \leftarrow \theta +\alpha\nabla_{\theta} J(\pi_\theta)
$$

因此，问题就转化为如何对我们目前的目标函数$J(\theta)$进行求导，也就是如何求解策略梯度$\nabla_\theta J(\pi_\theta)$

$$
\nabla_\theta J(\pi_\theta) = \mathbb{E}_{\tau \sim\pi_{\theta}}[R(\tau)\nabla_\theta \log p(\tau|\theta)] 
$$

上面的式子中$\pi_\theta(a_t|s_t)$代表智能体在t时刻选择动作$a_t$的概率，$R_t(\tau)$之前已经定义过了是指一条轨迹从t时刻开始的累计折旧回报。也就是说，这个式子表示目标函数的梯度计算可以通过指定的参数化策略网络$\pi_\theta$采样出很多条轨迹$\tau_1,\cdots,\tau_n$，并对每条轨迹计算它逐一时刻t的累计折旧回报和在当前状态$s_t$下采取$a_t$的对数概率的导数并累加。

这也就解释了为什么策略所产生的动作会随着策略梯度对参数的更新发生变化，因为如果在t时刻收益$R_t(\tau)>0$，那么行动$\pi_\theta(a_t | s_t)$的概率将会增加；如果在t时刻收益$R_t(\tau)>0$，那么行动$\pi_\theta(a_t | s_t)$的概率将会降低，这样经过多轮更新后模型将会学习到能生成更大激励$R_t(\tau)$的行为

这是因为$R_t(\tau)$ 越大时会让梯度向着获取更大收益的正向激励更新，当$R_t(\tau)$ 很小或者甚至为负时，那么则会让这个梯度方向的更新衰减甚至改变更新的方向，让模型向远离获得负收益的梯度方向更新，可以理解为激励可以作为梯度更新的权重。**对于监督学习，我们认为样本对应的标签是绝对正确的，所以我们默认在每次按照相同的权重向着标签的方向更新，而对于强化学习，我们不知道策略$\pi$所采样的动作是否是正确的，因此我们需要通过激励来控制我们梯度更新的方向和尺度。**
![[Untitled.png|1000]]
```ad-note
**左边**：一个高斯分布，蓝色的点是从分布中取样的动作样本。每个蓝色的点上都有一个蓝色的箭头指向一定的方向，这些方向是对数概率的梯度反向（相对于高斯分布的平均参数），沿着这些方向移动分布，可以提高该样本的概率。
**中间**：用得分函数覆盖整个分布，基本上到处都是-1，除了在一些很小的局部区域是+1（注意这个可以是随机的，并不必须是可微的标量函数）。现在箭头的颜色不同了，这是因为在更新的乘法中，我们将会平均所有的绿色箭头，以及红色箭头的负值。
**右边**：在参数更新之后，绿色箭头和反向的红色箭头推动着我们分布向着左边和下边方向移动。从现在这个分布中取样的动作样本将有有一个更高的期望得分，这和我们想要的是一致的。
```
对于目标函数的倒数可以写成如下形式：

$$
\nabla_\theta J(\pi_\theta) = \nabla_\theta\mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)]
$$

但是对于这种形式，我们已知参数  
$\theta$ 无法对 $R(\tau) = \sum_{t=0}^T\gamma^tr_t$ 进行求导，因为$r_t$是由一个未知的奖励函数$R(s_t,a_t,s_{t+1})$决定的，对于 $\theta$ 来说唯一能够影响$R(\tau)$的方式就是改变通过更新策略改变状态和动作的分布，从而影响智能体收到的激励。

因此，我们可以形式化的定义如下公式：

$$
\nabla_\theta \mathbb{E}_{x\sim p(x|\theta)}[f(x)]
$$

类似之前的目标函数，此处的x是在一个条件概率分布$p(x|\theta)$下采样得到的值，而$f()$是关于x的函数，我们需要做的事情是对这样一个期望关于参数 $\theta$ 求导：

$$
\nabla_\theta \int dxf(x)p(x|\theta) \\
=\int dx \nabla_\theta(p(x|\theta)f(x))\\
=\int dx(\nabla_\theta p(x|\theta)f(x)+\nabla_\theta f(x)p(x|\theta)) \\
= \int\nabla_\theta p(x|\theta)f(x)dx \\
=\int dx \frac{\nabla_\theta p(x|\theta)}{p(x|\theta)}\cdot p(x|\theta)f(x)\\
= \int dx \nabla_\theta\log p(x|\theta)p(x|\theta)f(x)\\
=\mathbb{E}_x[f(x)\nabla_\theta\log p(x|\theta)]
$$

以上公式推导证明了原期望的导数等价于对数概率的梯度乘以原函数的期望，这样我们就可以形式化的推导出$\mathbb{E}[f(x)]$的梯度。带入到原问题中，令$f(x) = R(\tau)$则可得：

$$
\mathbb{E}_{\tau\sim\theta} [R(\tau)] = \mathbb{E}_{\tau\sim\pi_\theta}[\nabla \log p(\tau|\theta)R(\tau)]
$$

对于一条轨迹$\tau$，是由一系列的状态和动作组成的序列，因此$p(\tau|\theta)$可以表示为：

$$
p(\tau|\theta) = \prod_{t\geq 0}p(s_{t+1}|s_t,a_t)\pi_\theta(a_t|s_t)
$$

其中$p(s_{t+1}|s_t,a_t)$ 代表在$s_t$下采取动作$a_t$到$s_{s+1}$的状态转移概率，$\pi_\theta(a_t|s_t)$代表策略$\pi_\theta$在状态$s_t$下采取动作$a_t$的概率。

$$
\log p(\tau|\theta) = \sum_{t \geq 0}\log p(s_{t+1}|s_t, a_t)+\sum_{t \geq 0}\pi_\theta(a_t|s_t) \\
\nabla_\theta\log p(\tau|\theta) =\sum_{t \geq 0}\nabla_\theta \log p(s_{t+1}|a_t,s_t) +\sum_{t \geq 0} \log \nabla_\theta \pi_\theta(a_t|s_t)\\
\nabla_\theta \log p(\tau|\theta) = \sum_{t\geq 0}\nabla_\theta\log \pi_{\theta}(a_t|s_t)
$$

同时我们也易得：$R(\tau) = \sum_{t \geq 0} R_t(\tau)$，所以可以将梯度公式重写为：

$$
\nabla_\theta J(\pi_\theta)= \mathbb{E}_{\tau \sim \pi_\theta}[\sum_{t=0}^{T}R_t(\tau)\cdot\sum_{t\geq 0}\nabla_\theta\log \pi_{\theta}(a_t|s_t)]
$$

并可以等价写作：

$$
\nabla_\theta J(\pi_\theta)= \mathbb{E}_{\tau \sim \pi_\theta}[\sum_{t=0}^{T}R_t(\tau)\nabla_\theta\log \pi_{\theta}(a_t|s_t)]
$$

对于这种等价性存在疑惑，可以参考这篇文章的数学证明。

[策略梯度推导 | To be a legend](https://www.yihaosun.cn/2021/04/02/%E7%AD%96%E7%95%A5%E6%A2%AF%E5%BA%A6%E6%8E%A8%E5%AF%BC/)

虽然已经形式化的推导出了目标函数的梯度公式，但对于计算来说仍然需要采样多条轨迹求取平均作为期望的估计，为了降低计算的复杂度，我们可以通过Monte-Carlo采样从中随机抽取一条轨迹$\tau$进行计算，这样做的合理性在于monte-carlo随机采样的轨迹计算出的目标函数梯度是真实期望的无偏估计：

$$
\nabla_\theta J(\pi_\theta) \approx \sum_{t=0}^{T}R_t(\tau)\nabla_\theta\log \pi_{\theta}(a_t|s_t)
$$

### 5. 算法流程

朴素的REINFORCE算法非常简单，首先初始化学习率$\alpha$和神经网络的参数$\theta$，对于每一轮训练随机采样一条轨迹$\tau$，然后在轨迹的每一个时间步t更新梯度
![[Untitled 1.png|1000]]
值得注意的是REINFORCE算法是一种on-policy算法，这意味着每一轮训练所采样得到的轨迹数据都无法重复用于网络的训练。在之前我们曾提到过对于同一神经网络$\pi$，$\pi_{\theta_1}$和$\pi_{\theta_2}$实际代表完全不同的策略，对于策略$\pi_{\theta_2}$来说它不可能采样出和$\pi_{\theta_1}$一样的轨迹，因此$\pi_{\theta_1}$所采样的轨迹无法用于更新$\pi_{\theta_2}$的梯度。虽然Monte-Carlo采样可以保证策略梯度估计的无偏性，但毫无疑问这是一个高偏差的结果，因为采样得到的不同轨迹，其返回的return也千变万化。（机器学习中曾学习过Bias-Varience trade-off）

- action采样时存在随机性
- 每个episode开始的状态存在随机性
- 环境转移函数可以是随机的

针对这一问题，一种可以削减策略梯度估计方差的改进是让$t$时刻的累计折旧回报减去一个与action无关的baseline，可以表示为：

$$
\nabla_\theta J(\pi_\theta) \approx \sum_{t=0}^{T}(R_t(\tau)-b(s_t))\nabla_\theta\log \pi_{\theta}(a_t|s_t)
$$

通常一般用值函数$V^\pi$来表示这个基线，这种方法用于A2C算法中（另一种强化学习算法）；另一种方法是认为$b = \frac{1}{T}\sum_{t=0}^{T}R_t(\tau)$，这其实是一个同化数据分布的操作（类似于BatchNorm或者是LayerNorm），它使得不同轨迹的$R_t(\tau)$零中心化。这样做有什么好处呢，考虑一个Return全为负的environment，这是即使你的策略采样了一个很好的action，它的returns仍然是负的，我们之前讲过returns的意义在于帮助我们的神经网络向returns高的方向更多的更新梯度，如果returns一直是负的，随着时间的推移也总能学习到更好的action，因为returns之间存在着相对大小，但这会导致收敛速度很慢，所以只有当returns是一个零中心的分布（有正有负），模型收敛速度才会更快。


