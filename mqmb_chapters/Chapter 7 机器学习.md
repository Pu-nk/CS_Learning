# Linear Model
## 线性回归
![[Pasted image 20230315101903.png]]
- 假设空间 $\mathcal{H}$ 是一个映射 $\mathcal{X} \rightarrow \mathcal{Y}$ 的函数的集合， 我们希望这些函数满足一些正则性（连续性、光滑性以及简单性）。假设空间是线性空间，可以用线性空间中如何去寻找一个超平面的方法来做；假设空间是树，可以采用树搜索算法来做
- 线性回归中的假设空间：
	- Training data: $\left(\boldsymbol{x}_1, y_1\right),\left(\boldsymbol{x}_2, y_2\right), \ldots,\left(\boldsymbol{x}_n, y_n\right)$， Feature vector: $\boldsymbol{x} \in \mathbb{R}^d$, response: $y \in \mathbb{R}$
	- Prediction of hypothesis $h$ parametrized by $\boldsymbol{w}$ :
$$
h(\boldsymbol{x})={w_0}+\sum_{j=1}^d w_j x_j=\sum_{j=0}^d w_j x_j=w \cdot \boldsymbol{x}
$$
![[Pasted image 20230315103733.png]]
- 损失函数
	- l2 损失函数具有光滑性，优化简单
![[Pasted image 20230315105545.png]]
- 线性回归具有解析解（计算速度快），但计算速度其实并不快，算法复杂度$O(d^2(n+d))$
- 梯度下降：
$$
J(\boldsymbol{w})=J\left(\boldsymbol{w}_0\right)+\left(\boldsymbol{w}-\boldsymbol{w}_0\right)^T \boldsymbol{g}+\cdots 
$$
$$

J(\boldsymbol{w}-\eta \boldsymbol{g}) \approx J(\boldsymbol{w})-\eta \boldsymbol{g}^T \boldsymbol{g}
$$
也就是说只要沿着梯度方向调整权重，就会使得损失函数减小，算法复杂度为$O(ndT)$, 如果n 特别大会采用随机梯度下降（SGD），算法复杂度为$O(mdT)$， 对于维度爆炸的问题可以采用并行化、分布式计算的方法来解决

```ad-tip
对n可以采用因为计算损失函数是计算所有样本上损失函数的和，采样m是n的无偏估计，而对于维度d来说，采样其中的部分维度相当于是进入了原问题的一个线性子空间，会丢失信息。

```

## 非线性化
## 正则化
## 线性分类
# Decision Tree
tabular特征/属性具有异构性构建线性模型（做归一化、0-1encoding）
希望树模型具有可解释性、泛化性
## 决策树的挑战
- 机器如果找到最有用的特征
- 什么时候停止树的生长
- 如何处理缺失值
## ID3 算法
ID3：只解决了节点分裂的问题
一种视角是Node splitting代表的是一种分组方式，在每次划分中需要找到最好的那个划分。但如何定义分类结果好？是数量平衡还是每一类的纯粹度更高？下面是一些用于度量分类纯粹度的指标：
- **分类错误率**：$\min\left( \frac{|\mathcal{c}_{1}|}{|\mathcal{D}|}, \frac{|\mathcal{c}_{2}|}{|\mathcal{D}|} \right)$， 对于k类，应该取出最多的那一类然后计算$\min_{1\leq i\leq k}\left( \frac{|\mathcal{c}_{k}|}{|\mathcal{D}|} \right)$，分类错误率会导致不同分割的结果是一样的,因为线性很容易共线。
- **熵函数**：$-\sum_{k=1}^{K}\frac{|c_{k}|}{|D|}\log{\frac{|c_{k}|}{|D|}}$，评价分类结果的不确定性。
- **基尼系数**：$1 - \sum_{k=1}^{K}\left( \frac{|c_{k}|}{|D|} \right)^2$
![[Pasted image 20230118101459.png|800]]
如何度量一个划分的好坏，直观的想法是对比父节点的熵和子节点的熵的平均值，但两个子节点的大小不一定相等，为了解决这个问题可以用两个子节点的样本比例的加权和。
$$
\frac{|D_{1}|}{|D|}H(D_{1}) + \frac{|D_{2}|}{|D|}H(D_{2})
$$
信息增益IG = 父节点熵 - 子节点熵的加权和
分割搜索（split search）:计算所有划分对应的信息增益，找出对分类效果最好的特征。
**ID3算法流程**
- 找到根节点最好的分类特征
- 将根节点数据分到对应的子节点
- 迭代以上过程直到满足下面三种停止条件
	- 分类全是正例
	- 分类全是负例
	- 没有可用的特征，返回节点对应数据集中的分类投票结果
```ad-note
title: 特别状况
如果分类后的某个子节点为空，会导致对未来数据无法进行分类，因此会使用全体数据的最大分类作为该节点分类结果
```
## C4.5 算法
- 节点分裂准则
- 解决特征多样性的问题
- 解决决策树的过拟合问题
### 节点分裂准则
信息增益对于多值的特征是有偏的，它会偏向那些属性取值特别多的属性，但这个对于分类来说并没有什么意义，所以提出了一个gain ratio的东西：
$$
GR = \frac{IG}{IV}
$$
其中$IV$代表一个惩罚项，用于度量分裂后的均衡程度
$$
IV(f) = -\sum_{i=1}^{|V|}\frac{|\mathcal{D}_{i}|}{|\mathcal{D}|}\log\frac{|\mathcal{D_{i}}|}{|\mathcal{D}|}
$$
同时，为了人为的去设定哪个特征重要，也可以设定代价敏感的节点分裂函数
$$
\frac{GR^2}{Cost}
$$
这样可以手动的定义不同特征的cost从而改变分裂路径，将领域先验知识融合
### 处理缺失值
如果将IG适配到有缺失值的情况：
- 如何计算有缺失值的IG: $(1-\rho)\cdot IG$
- 如何将数据assign到子节点熵：将含有缺失值的样本单独分到一类，然后在下一次分裂之前中将这些样本按照比例分到兄弟节点中
![[Pasted image 20230118112013.png]]
### 处理连续值
增加阈值来对数据进行切分，这样每个节点都只有两个子节点
$$
\{x|x_{j}\leq s\} \cup \{x|x_{j} > s\} 
$$
将连续值的样本点进行排序，然后以任意两个点的中间点作为切分点
$$
s_{j} \in \left\{ \frac{1}{2}(x_{rj} + x_{(r+1)j}) |r=1,\cdots, n-1 \right\}
$$
![[Pasted image 20230118113134.png]]
搜索每个切分点并从里面找到信息增益最高的切分点作为最佳切分点。
### 解决过拟合问题
- 预剪枝(pre-pruning):在分裂之后先算一遍验证集的误差，如果分裂之后验证误差变低了（可以设置阈值），那么在验证集上这个分裂是有效的，否则就不做这个分裂，但预剪枝存在的问题是因为某一次分裂可能在验证集上没有增益，但是对于未来的数据有用（过于贪婪的剪枝方法）
- 后剪枝(post-pruning):从叶节点开始，对比剪枝前后的效果，如果剪枝后效果更好说明过拟合了。
- 复杂度正则项：计算剪枝前后的误差如果$C_{alpha}(T_{A}) < C_{alpha}(T_{B})$
$$
\begin{align}
C_{\alpha}(T) & = \hat{\epsilon}(T) + \alpha\mid T\mid  \\
& = \sum_{t=1}^{\mid T\mid}N_{t}H_{t}(T) + \alpha\mid T\mid  \\
& = \sum_{t=1}^{\mid T\mid}\sum_{k=1}^{K} N_{tk}\log \frac{N_{tk}}{N_{t}} + \alpha\mid T\mid
\end{align}


$$
```ad-note
title: 决策树损失函数
决策树的损失函数就是所有叶子节点熵函数的加权和，不直接优化损失函数，而是通过贪心的方式选择最优的特征来间接优化目标函数
```
## CART算法
CART是根据某个特征是否等于某个值将它分成一个二叉树
- 分类树
	- 使用基尼系数作为节点分裂的评价（熵最小化学到的伪标记某个label接近于1，在其他label接近于0，用基尼指数就不会这么极端）
$$
\operatorname{Gini}(\mathcal{D}, A)=\frac{\left|\mathcal{D}_1\right|}{|\mathcal{D}|} \operatorname{Gini}\left(\mathcal{D}_1\right)+\frac{\left|\mathcal{D}_2\right|}{|\mathcal{D}|} \operatorname{Gini}\left(\mathcal{D}_2\right)
$$
- 回归树
	- 给一个region赋一个目标值，这个region里所有y的平均值
	- 在回归中最小化左右子树的$L_{2}$损失函数
$$\widehat{v}_k=\operatorname{ave}\left(y_i \mid x_i \in R_k\right)$$
![[Pasted image 20230118125033.png]]
## 决策树小结
-  决策树的分类空间是一些平行于坐标轴的空间划分。
![[Pasted image 20230118123357.png]]
- 实际上是在做非线性特征提取，决策树的本质是通过空间的分区将原始的数据变成一个新的特征向量，然后在这个向量上做线性分类，决策树不是可微的，不能做端到端的训练
$$
h(x) = \sum_{i=1}^{m}c_{i}\mathrm{1}\{x \in R_{i}\}
$$
	这个$\mathrm{1}\{x \in R_{i}\}$其实可以看作是一个特征向量，而决策树其实是在做一个特征变换：
	$x \rightarrow \Phi(x) = (1\{x\in R_{1}\}), \cdots, 1\{x\in R_{m}\})$
- 多元决策树，在邻域做拟合，生成不平行的分类边界（表达能力上升，可解释性下降）
![[Pasted image 20230118122252.png]]
- 泛化界
	- ...
- 决策树是可解释的，但是它是不stable的（可解释性有意义吗？）
- 决策树是非几何性的，没有采用任何度量所以可以特征的尺度没有影响
- 决策树可以同时处理分类变量和连续变量
- 决策树可以处理缺失值和噪声数据
- 决策树的预测函数是非连续的
# Ensemble Learning
- 并行的训练多个分类器，并通过多数投票作为集成输出
$$
H(x) = \text{sign}\left( \sum_{t=1}^{T}hT(x) \right)
$$
- 假设每个基学习器的期望误差为$\epsilon: P(h_{t}(x) \neq y)$，我们希望控制总误差$H(x)$：
$$
P(H(x) \neq y) = P\left( \sum_{t=1}^{T} \mathbf{1}\left[h_{t}(x) \neq y \right] \geq \frac{1}{2} \times T\right)
$$
- 集成学习误差上界:对于相互独立的学习器，集成学习误差随着学习器的增加呈指数衰减
$$
P\left( H(x) \neq y\right) \leq 2\exp\left(-\frac{1}{2}T(1 - 2\epsilon)\right) 
$$
那么，如何去增强学习器之间的多样性
- Bootstrap：随机筛选样本
- Random Forests：随机筛选特征
- Boosting: 对于不同样本的加权
## 集成学习类型
树模型稳定性比较差，如何降低方差
- Parallel ensemble
	- 并行构建模型
	- 构建多个学习器来避免过拟合，降低方差
- Sequential emsemble
	- 串行构建模型
	- 后面的模型都感知到了前面的模型，可以做针对性的改进，降低偏差
## 随机森林 
本质上是一个多学习器的模型，构造多个子数据集，训练多个子模型，进行投票（均值）
自助采样（Bootstrap）：有放回的采样n次，得到n条样本构成一个子数据集
某条样本不出现的概率：$\left( 1-\frac{1}{n} \right)^n \rightarrow \frac{1}{e}$，有63%的数据至少出现了一次，这保证了采样得到的数据与原始数据集不一致
Bootstrap Aggregation(Bagging): 
- 非常通用的降低方差的方法
- OOB预测误差就是天然的验证集误差
随机森林：
	1. bootstrap 构建b个数据集
	2. 随机选择k个特征子集（$k = \sqrt{ d }$）,使用比较少的特征防止过拟合
	3. 用线性模型做随机森林不太合适，因为线性模型本身方差很小，用来降低方差没有收益
meta overfitting （训练集和验证集都过拟合了）
## Adaboost
Weak Learner: 在一个假设空间中存在一个弱学习算法，其期望误差接近$\frac{1}{2}$(只要比随机猜好一点)
$$
P_{\mathcal{D_{n}}\sim D^n}\left[\epsilon\left( h_{\mathcal(D_{n})} \leq \frac{1}{2} -\gamma \right)\right] \geq 1-\sigma
$$
在多项式的样本条数的基础上学习出接近$\frac{1}{2}$的分类误差结果才可以被认为是弱分类器，因为在无穷多样本上学习到这一结果没有意义。
经典的弱学习器
- 决策树桩：只有一个根节点和两个叶子节点（二分类器）
- 回归树桩：决策树桩的回归版本
Boost的思想是找到一些弱学习器，如何将这些弱学习器进行提升，将其变成强学习器
![[Pasted image 20230128112021.png | 800]]
$$
\begin{align}
& f_{t}(x) = \sum_{s=1}^t \alpha_{s}h_{s}(x)  \\
& h(x) = \text{sgn}(f_{t}(x))
\end{align}
$$
![[Pasted image 20230128112336.png| 800]]
形式化定义：集成学习模型$h(x)$的优化问题实际是找到t个基模型$h_{s}(x)$ 和权重$\alpha_{s}$，使得：
$$
(\alpha_{1},\dots,\alpha_{t},h_{1},\cdots, h_{t}) = \argmin_{(\alpha_{1},\dots,\alpha_{t},h_{1},\cdots, h_{t})} \frac{1}{n}\sum_{i=1}^{n}\mathcal{l}\left(y_{i}, \sum_{s=1}^t\alpha_{s}h_{s}(x_{i})\right)
$$
联合优化2t个参数是复杂的,因此Boosting直接采用了串行优化的方式来构建模型
AdaBoost:我们假设上一个模型已经是训练好的，那么每次我们只需要优化一步
$$
(\alpha_{t},h_{t}) = \argmin_{(\alpha_{t},h_{t})} \frac{1}{n} \sum_{i=1}^{n}\mathcal{l}(y_{i}, f_{t-1}(x_{i} + \alpha_{t} h_{t}(x_{i})))
$$
![[Pasted image 20230128114235.png | 400]]
```ad-note
title: 分类损失函数
0 - 1 损失函数是分类函数的黄金准则，经常使用的损失函数都是0-1损失函数的凸上界，这样的损失函数称为代理损失函数(surrogate loss function)，实际中去试图利用这些函数逼近0-1损失函数的解，如果解一致称为一致性损失函数，优化问题不一样造成的误差（优化误差）
```
adaboost使用的损失函数 - 指数损失函数 $\mathcal{l}(y, f(x)) = e^{-yf(x)}$
```ad-question
title: 为什么这里可以用指数函数
指数函数梯度太大，而且对噪音不鲁棒，但是指数函数是0-1损失函数的一致性损失函数,同时对于一个指数损失函数令它期望误差的导数为0，其决策函数是一个logistic线性函数
$$
f(x) = \frac{1}{2}\log{\frac{P(y=1|x)}{P(y=-1|x)}}
$$
可以发现$f(x)$和贝叶斯分类器分类结果一样，所以指数函数替代0-1损失函数是等效的
```
Adaboost 算法过程
- 每一条样本分配初始权重$\frac{1}{n}(\mathcal{D_{1}})$
-  迭代1...T个学习器
	- $\alpha_{t} = \frac{1}{2}\log{\frac{1-\epsilon_{t}}{\epsilon_{t}}}$，分类器的分类误差越小，这个学习器的权重就越大
	- $Z_{t} = 2[\epsilon_{t}(1-\epsilon_{t})]^{1/2}$,归一化因子，使得$D_{t}$是一个概率分布
	- 按照如下规则迭代更新每条样本的权重
$$
D_{t+1}(i) = \frac{D_{t}(i)\exp(-\alpha_{t}y_{i}h_{t}(x_{i}))}{Z_{t}}=\frac{D_{t-1}(i)\exp(-\alpha_{t-1}y_{i}h_{t-1}(x_{i}))\exp(-\alpha_{t}y_{i}h_{t}(x_{i}))}{Z_{t-1}Z_{t}}=\frac{1}{n} \frac{\exp(-y_{i}f_{t}(x_{i}))}{\prod_{s=1}^tZ_{s}}
$$
指数损失函数优化Boosting的目标函数
$$
F(\bar{\alpha}) = \frac{1}{n} \sum_{i=1}^n e^{-y_{i}f(x_{i})} = \frac{1}{n}\sum_{i=1}^ne^{-y_{i}\sum_{j=1}^{N}\bar{\alpha}_{j}h_{j}(x_{i})}
$$
其中$\bar{\alpha}$代表联合优化得到的权重，而非贪心串行计算得到的
- 先找到导数方向最大的学习器进行优化，等价于找到训练误差最小的基模型
$$
F(\bar{\alpha}_{t-1} + \eta e_{k}) = \frac{1}{n} \sum_{i=1}^n e^{-y_i \sum_{j=1}^N \bar{\alpha}_{t-1, j} h_j\left(x_i\right)-\eta y_i h_k\left(x_i\right)}
$$
$$
\begin{aligned}
F^{\prime}\left(\overline{\boldsymbol{\alpha}}_{t-1}, \boldsymbol{e}_k\right) & =\lim _{\eta \rightarrow 0} \frac{F\left(\overline{\boldsymbol{\alpha}}_{t-1}+\eta \boldsymbol{e}_k\right)-F\left(\overline{\boldsymbol{\alpha}}_{t-1}\right)}{\eta} \\
& =-\frac{1}{n} \sum_{i=1}^n y_i h_k\left(\boldsymbol{x}_i\right) e^{-y_i \sum_{j=1}^N \bar{\alpha}_{t-1, j} h_j\left(x_i\right)} \\
& =-\frac{1}{n} \sum_{i=1}^n y_i h_k\left(\boldsymbol{x}_i\right) \bar{D}_t(i) \bar{Z}_t
\\
& =-\left[\left(1-\bar{\epsilon}_{t, k}\right)-\bar{\epsilon}_{t, k}\right] \frac{\bar{Z}_t}{n}\\
& =\left[2 \bar{\epsilon}_{t, k}-1\right] \frac{\bar{Z}_t}{n}
\end{aligned}
$$
- 在该方向上对步长进行优化，线搜索
$$
\begin{aligned}
&\frac{\partial F\left(\overline{\boldsymbol{\alpha}}_{t-1}+\eta \boldsymbol{e}_k\right)}{\partial \eta}=0 \Leftrightarrow-\frac{1}{n} \sum_{i=1}^n y_i h_k\left(\boldsymbol{x}_i\right) e^{-y_i \sum_{j=1}^N \bar{\alpha}_{t-1, j} h_j\left(x_i\right)-\eta y_i h_k\left(\boldsymbol{x}_i\right)}=0 \\

&\Leftrightarrow \sum_{i=1}^n y_i h_k\left(\boldsymbol{x}_i\right) \bar{D}_t(i) \bar{Z}_t e^{-\eta y_i h_k\left(\boldsymbol{x}_i\right)}=0 \\

&\Leftrightarrow \sum_{i=1}^n \bar{D}_t(i) y_i h_k\left(\boldsymbol{x}_i\right) e^{-\eta y_i h_k\left(\boldsymbol{x}_i\right)}=0 \\

& \Leftrightarrow \sum_{i=1}^n \bar{D}_t(i) \mathbf{1}_{\left[y_i h_k\left(\boldsymbol{x}_i\right)=+1\right]} e^{-\eta}-\sum_{i=1}^n \bar{D}_t(i) \mathbf{1}_{\left[y_i h_k\left(x_i\right)=-1\right]} e^\eta=0 \\
& \Leftrightarrow\left[\left(1-\bar{\epsilon}_{t, k}\right) e^{-\eta}-\bar{\epsilon}_{t, k} e^\eta\right]=0 \\
& \Leftrightarrow \eta=\frac{1}{2} \log \frac{1-\bar{\epsilon}_{t, k}}{\bar{\epsilon}_{t, k}}
\end{aligned}
$$
学习过程中间隔不断增加，误差不断变小
## Gradient Boosting
AdaBoost的缺点：
- 只针对分类问题和指数损失函数
- 容易受到噪声数据的干扰
Boosting的目标函数可以被定义为如下形式，其中$u_{t}$代表$\alpha_{t}h_{t}$，$v_{t}$代表$f(t-1)$：
$$
F\left(\boldsymbol{u}_t, \boldsymbol{v}_t\right) \triangleq \frac{1}{n} \sum_{i=1}^n \ell\left(y_i, u_{t i}+v_{t i}\right)
$$
所以要解决的问题其实是如何选择$u_{t}$，使得目标函数尽量小。不妨先假定$u_{t}$的已知，那么其实就是一个关于$v_{t}$的函数。朝着$v_{t}$的方向的导数更新一个步长，如果$v_{t}$可以变，那么更新之后的$v_{t}$必然会使目标函数下降
$$
\boldsymbol{v}_t-\eta \nabla_{\boldsymbol{v}_t} \frac{1}{n} \sum_{i=1}^n \ell\left(y_i, v_{t i}\right)
$$
对比更新后的式子与原来的公式，$v_{t}$是不变的也就是说如果希望目标函数变小，只要每次让$u_{t}$逼近$v_{t}$的梯度就好了
$$
\frac{1}{n} \sum_{i=1}^n \ell\left(y_i, v_{t i}-\eta \nabla_{v_{t i}} \frac{1}{n} \sum_{i=1}^n \ell\left(y_i, v_{t i}\right)\right)
$$
得到gradient boosting的框架：
$$
h_t\left(\boldsymbol{x}_i\right) \approx-\nabla_{v_{t i}} \frac{1}{n} \sum_{i=1}^n \ell\left(y_i, v_{t i}\right)=-\nabla_{v_{t i}} \ell\left(y_i, v_{t i}\right)
$$
这说明 Gradient Boosting 是通过新增加一个基模型来逼近梯度，同时这个基模型需要是连续的
- Gradient Descent: $\theta_{t} - \eta\nabla_{\theta_{t}}l$
- Gradient Boosting: 新添加一个模型来逼近梯度下降的梯度
GBDT 流程：
- 计算t-1轮的加权模型$f(t-1)$
- 令$h_{t}$等于$h_t\left(\boldsymbol{x}_i\right) \approx-\nabla_{v_{t i}} \frac{1}{n} \sum_{i=1}^n \ell\left(y_i, v_{t i}\right)=-\nabla_{v_{t i}} \ell\left(y_i, v_{t i}\right)$
- 学习率$\alpha_{t}>0$
Gradient Boosting只要求损失函数对模型可微，并不要求模型对参数可微

L2 Boosting
采用L2损失函数作为gradient boosting的损失函数进行回归预测
- 指定损失函数
$$
\begin{gathered}
\ell\left(f_{t-1}\right)=\sum_{i=1}^n\left(y_i-f_{t-1}\left(\boldsymbol{x}_i\right)\right)^2 \\
\ell\left(\boldsymbol{v}_t\right)=\sum_{i=1}^n\left(y_i-v_{t i}\right)^2
\end{gathered}
$$
- 计算梯度
$$
-\partial_{v_{t i}} \ell\left(\boldsymbol{v}_t\right)=-\partial_{v_{t i}}\left(v_{t i}-y_i\right)^2=2\left(y_i-v_{t i}\right)=2\left(y_i-f_{t-1}\left(\boldsymbol{x}_i\right)\right)
$$
```ad-note
一般不会用神经网络来作为基模型进行boosting，因为boosting主要是用于降低模型的偏差，而神经网络本身是一个强学习器，偏差较小。
```
- 优化问题
$$
h_t=\operatorname{argmin}_h \sum_{i=1}^n\left(h\left(\boldsymbol{x}_i\right)-2\left(y_i-f_{t-1}\left(\boldsymbol{x}_i\right)\right)\right)^2
$$
Binomial Boosting
- 指定损失函数
$$
\ell(y, f(x))=\log \left(1+e^{-y f(x)}\right)
$$
- 计算梯度
$$
-\partial_{v_{t i}}\left[\log \left(1+e^{-y_i v_{t i}}\right)\right]=\frac{y_i e^{-y_i v_{t i}}}{1+e^{-y_i v_{t i}}}=\frac{y_i}{1+e^{y_i v_{t i}}}
$$
- 优化问题
$$
h_t=\operatorname{argmin}_h \sum_{i=1}^n\left(h\left(\boldsymbol{x}_i\right)-\frac{y_i}{1+e^{y_i f_{t-1}\left(x_i\right)}}\right)^2
$$
## XGBoost
XGBoost主要创新点在于对GBDT增加正则项
- 并行化
- 分布式计算
- 缓存优化
XGBoost的正则项定义为叶子节点的数量和叶子权重的二范数
$$
\Omega\left(f_t\right)=\gamma T+\frac{1}{2} \lambda \sum_{i=1}^T w_j^2
$$

XGBoost的损失函数定义如下所示：
$$
\begin{aligned}
\operatorname{Obj}^{(t)} & =\sum_{i=1}^n \ell\left(y_i, \widehat{y}_i^{(t)}\right)+\sum_{k=1}^t \Omega\left(f_t\right) \\
& =\sum_{i=1}^n \ell\left(y_i, \widehat{y}_i^{(t-1)}+f_t\left(\boldsymbol{x}_i\right)\right)+\Omega\left(f_t\right)+\mathrm{const}
\end{aligned}
$$
二阶泰勒展开，损失函数表示为：
$$
\mathrm{Obj}^{(t)} \approx \sum_{i=1}^n\left[g_i f_t\left(\boldsymbol{x}_i\right)+\frac{1}{2} h_i f_t^2\left(\boldsymbol{x}_i\right)\right]+\Omega\left(f_t\right)
$$
$$
g_i=\partial_{\hat{y}^{(t-1)}} \ell\left(y_i, \hat{y}^{(t-1)}\right), h_i=\partial_{\hat{y}^{(t-1)}}^2 \ell\left(y_i, \hat{y}^{(t-1)}\right)
$$
推导目标函数梯度
$$
\begin{aligned}
\operatorname{Obj}^{(t)} & \approx \sum_{i=1}^n\left[g_i f_t\left(\boldsymbol{x}_i\right)+\frac{1}{2} h_i f_t^2\left(\boldsymbol{x}_i\right)\right]+\Omega\left(f_t\right) \\
& =\sum_{i=1}^n\left[g_i w_{q\left(x_i\right)}+\frac{1}{2} h_i w_{q\left(x_i\right)}^2\right]+\gamma T+\frac{1}{2} \lambda \sum_{i=1}^T w_j^2 \\
& =\sum_{j=1}^T\left[\left(\sum_{i \in I_j} g_i\right) w_j+\frac{1}{2}\left(\sum_{i \in I_j} h_i+\lambda\right) w_j^2\right]+\gamma T
\end{aligned}
$$
这样推导的意义在于能够对于不同叶子节点的解耦，其中$G$和$H$都可以通过计算目标函数对模型的一次导数和二次导数并代入上一步预测值计算
$$
w_j^*=-\frac{G_j}{H_j+\lambda}, \quad \text { Obj }=-\frac{1}{2} \sum_{j=1}^T \frac{G_j^2}{H_j+\lambda}+\gamma T
$$

$$
\text{Gain}=\frac{G_L^2}{H_L+\lambda}+\frac{G_R^2}{H_R+\lambda}-\frac{\left(G_L+G_R\right)^2}{H_L+H_R+\lambda}-\gamma
$$
# 聚类
## 相似度量
- 欧式距离 $\mathrm{D}(\boldsymbol{x}, \boldsymbol{z})=\|\boldsymbol{x}-\boldsymbol{z}\|_2=\sqrt[2]{\left(\sum_{j=1}^d\left(x_j-z_j\right)^2\right)}$
- 闵可夫斯基距离 $\mathrm{D}(\boldsymbol{x}, \mathbf{z})=\|\boldsymbol{x}-\boldsymbol{z}\|_p=\sqrt[p]{\left(\sum_{j=1}^d\left(x_j-z_j\right)^p\right)}$
- 核函数 $\mathrm{D}(\boldsymbol{x}, \mathbf{z})=\|\Phi(\boldsymbol{x})-\boldsymbol{\Phi}(\mathbf{z})\|=\sqrt{k(\boldsymbol{x}, \boldsymbol{x})+k(\mathbf{z}, \mathbf{z})-2 k(\boldsymbol{x}, \mathbf{z})}$
Kmeans假设了每个聚类中存在一个聚类簇中心（一个很强的假设）,使得每个点距离其聚类簇中心点和距离最近

$$
\min _{\boldsymbol{\mu}_1, \ldots \boldsymbol{\mu}_k} \sum_{i=1}^n\left\|\boldsymbol{\mu}_{h\left(\boldsymbol{x}_i\right)}-\boldsymbol{x}_i\right\|_2^2
$$
### Kmeans 步骤
- 初始化k个聚类中心
- 划分每一个点到对应的聚类中心（这样可以使得目标函数较小）
- 更新聚类中心为聚类簇的均值（同样可以使得目标函数较小，L2损失函数意义下的最小值对应聚类簇的均值）
```ad-question
title: 如果使用L1损失函数会怎样？
使用L1损失函数就是K中心算法，使用L1损失函数的好处在于它对于离群值不敏感（中位数）
```
### Kmeans 算法的收敛
聚类算法存在初值问题（所有的非凸问题都存在），为了解决Kmeans受到初值的影响，采用Kmeans++算法：
- 初始化第一个点 $\boldsymbol{\mu}_1 \longleftarrow \text { Random }\left(\boldsymbol{x}_{(1: n)}\right)$
- 按照特定的概率分布来寻找初始点 $\boldsymbol{\mu}_j \longleftarrow \text { Random }\left(\boldsymbol{x}_{(1: n)}, p_{(1: n)}\right)$， 其中 $p_i \propto \min _{j^{\prime}<j}\left\|\boldsymbol{\mu}_{j^{\prime}}-\boldsymbol{x}_i\right\|_2^2$，也就是说如果当前该点如果离已经分好的聚类中心比较近的话就会以比较小的概率被选中
如何选择聚类簇数量
- 观察loss的变化，选择合适的结果（loss小，但是聚类簇数量也不大）
- 信息准则$\mathrm{AIC}=2 k-2 \ln \widehat{L}$, $\mathrm{BIC}=k \ln n-2 \ln \widehat{L}$，选择使信息准则最优的
### Kmeans 做数据压缩
使用Kmeans做数据压缩（VQ，向量量化）或者词袋模型（BOW），特征学习是Kmeans的扩展
- 搜索系统：有一个数据库$\mathcal{X}=\left\{\boldsymbol{x}_1, \ldots, \boldsymbol{x}_n\right\}$和查询$\boldsymbol{q} \in \mathbb{R}^d$ ，找到$\mathrm{NN}(\boldsymbol{q})=\min _{\boldsymbol{x} \in X} \operatorname{dist}(\boldsymbol{x}, \boldsymbol{q})$
	- 缩小计算距离的次数的方法， $O\left(n^{\prime} d\right), n^{\prime} \ll n$
		- 索引技术 ，inverted index, search tree, neighborhood graph 
	- 缩小计算距离的损失， $O\left(n d^{\prime}\right), d^{\prime} \ll d$
		- 哈希技术， Locality- Sensitive Hashing(LSH), SPectral Hashing(SH)
		- 量化技术，将数据分成若干堆，然后在堆里进行搜索 $\operatorname{dist}(\boldsymbol{x}, \boldsymbol{q}) \approx \operatorname{dist}\left(\boldsymbol{\mu}_{h(\boldsymbol{x})}, \boldsymbol{\mu}_{h(\boldsymbol{q})}\right)$, 将任意两个点之间的距离近似成两个类簇中心的距离，作为固定的查找表加速，但是类簇数必须非常多，因为维度很高
		- 乘积量化：将数据分成m个子空间$\min _{\boldsymbol{\mu}_1^1, \ldots \boldsymbol{\mu}_k^m, y_1^1, \ldots, y_n^m} \sum_{j=1}^m \sum_{i=1}^n\left\|\boldsymbol{x}_i^j-\boldsymbol{\mu}_{y_i^j}^j\right\|_2^2$, 总距离就等于m个子空间距离之和 $\operatorname{dist}(\boldsymbol{x}, \boldsymbol{q})=\sum_{j=1}^m \operatorname{dist}\left(\boldsymbol{\mu}_{y_x^j}, \boldsymbol{\mu}_{y_{\boldsymbol{q}}^j}\right)$
### 聚类形态
Kmeans 很难对环状数据进行聚类
- 紧凑性- Kmeans
- 连通性 - 图聚类
### 图聚类
- Given data $\left\{\boldsymbol{x}_i\right\}_{i=1}^n$ and similarity $W\left(\boldsymbol{x}_i, \boldsymbol{x}_j\right)$ , partition data in groups: 
	- points in the same group are connected 
	- points in different groups are disconnected 
- Similarity Graph: $G(V, E, W)$  Vertices, Edge, Edge 
- 使用高斯核函数来具有特征向量的n条数据来构建图
- Weight (权重矩阵)
	- 邻接矩阵刻画连通性，使得聚类结果中分到同一个组的边权重之和比较大，而分到不同组的边的权重之和比较小
	- 割（Cut）：$\operatorname{cut}(A, B)=\sum_{i \in A, j \in B} W_{i j}$ A中的一个顶点和B中的顶点有边相连的权重之和，割函数实际刻画了两个子图的连通性。
	- Normalize Cut： $\operatorname{Ncut}(A, B)=\operatorname{cut}(A, B)\left(\frac{1}{\operatorname{vol}(A)}+\frac{1}{\operatorname{vol}(B)}\right)$
	- $\operatorname{vol}(A)=\sum_{i \in A} d_i$, $d_i=\sum_j W_{i j}$, 调和平均使得两个子集都不是特别小
- 图拉普拉斯矩阵
	- Let  $\boldsymbol{D}$  be the degree matrix: $\boldsymbol{D}(i, i)=\sum_j W_{i j}$
	- Unnormalized Graph Laplacian: 
		- $\boldsymbol{L}=\boldsymbol{D}-\boldsymbol{W}$
	- Normalized Graph Laplacian:  度矩阵变成单位阵是希望图上重要度一致
		- $\boldsymbol{D}$是对角阵，每个对角元为所有链接该点的权重的和
		- $\boldsymbol{L}_{\text {symmetric }}=\boldsymbol{D}^{-\frac{1}{2}} \boldsymbol{L} \boldsymbol{D}^{-\frac{1}{2}}=\boldsymbol{I}-\boldsymbol{D}^{-\frac{1}{2}} \boldsymbol{W} \boldsymbol{D}^{-\frac{1}{2}}$
		- $\boldsymbol{L}_{\text {RandomWalk }}=\boldsymbol{D}^{-1} \boldsymbol{L}=\boldsymbol{I}-\boldsymbol{D}^{-1} \boldsymbol{W}$ 每一列sum为1，类似概率分布，可以看作是一次在图上的随机游走
	- Normalized Cut
		- 图拉普拉斯矩阵是为了更有效率的计算Ncut
$$
f_i=\left\{\begin{array}{cc}
\frac{1}{\operatorname{vol}(A)} & \text { if } i \in A \\
-\frac{1}{\operatorname{vol}(B)} & \text { if } i \in B
\end{array}\right.
$$
				
$$
\operatorname{Ncut}(A, B)=\operatorname{cut}(A, B)\left(\frac{1}{\operatorname{vol}(A)}+\frac{1}{\operatorname{vol}(B)}\right)=\frac{\boldsymbol{f}^T \boldsymbol{L} \boldsymbol{f}}{\boldsymbol{f}^T \boldsymbol{D} \boldsymbol{f}}
$$
- 流形假设
$$
\boldsymbol{f}^T \boldsymbol{L} \boldsymbol{f}=\sum_{i, j} f_i L_{i j} f_j=\frac{1}{2} \sum_{i, j} W_{i j}\left(f_i-f_j\right)^2=\sum_{i \in A, j \in B} W_{i j}\left(\frac{1}{\operatorname{vol}(A)}+\frac{1}{\operatorname{vol}(B)}\right)^2
$$
$$
\boldsymbol{f}^T \boldsymbol{D} \boldsymbol{f}=\sum_i d_i f_i^2=\frac{1}{\operatorname{vol}(A)}+\frac{1}{\operatorname{vol}(B)}
$$
图正则化（流形假设）：如果在图相似的位置上，希望label也相似
$$
\min _{\boldsymbol{f}} \operatorname{Ncut}(A, B)=\frac{\boldsymbol{f}^T \boldsymbol{L} \boldsymbol{f}}{\boldsymbol{f}^T \boldsymbol{D} \boldsymbol{f}} \quad \text { s.t. } f_i \in\left\{\frac{1}{\operatorname{vol}(A)},-\frac{1}{\operatorname{vol}(B)}\right\}
$$
但是因为不能进行离散优化，将其变成连续形式优化：
$$
\min _{\boldsymbol{f}} \operatorname{Ncut}(A, B)=\frac{\boldsymbol{f}^T \boldsymbol{L} \boldsymbol{f}}{\boldsymbol{f}^T \boldsymbol{D} \boldsymbol{f}} \quad \text { s.t. } f_i \in \mathbb{R}
$$
瑞利商，两个二次型的商，这个问题等价于
$$
\min _{\boldsymbol{f}} \boldsymbol{f}^T \boldsymbol{L} \boldsymbol{f} \quad \text { s.t. } \quad \boldsymbol{f}^T \boldsymbol{D} \boldsymbol{f}=1
$$
 这个最优化问题的解就是这样一个广义特征值问题$L f=\lambda D f$,定义拉格朗日乘子法：
 $$
\mathcal{L}=\boldsymbol{f}^T \boldsymbol{L} \boldsymbol{f}-\lambda\left(\boldsymbol{f}^T \boldsymbol{D} \boldsymbol{f}-1\right) .
$$
$$
\boldsymbol{f}: \nabla_{\boldsymbol{f}} \mathcal{L}=2(\boldsymbol{L}-\lambda \boldsymbol{D}) \boldsymbol{f}=\mathbf{0} \Rightarrow \boldsymbol{L} \boldsymbol{f}=\lambda \boldsymbol{D} \boldsymbol{f}
$$
也就是说f是最优值的时候对应的二次型的值是特征值$\lambda$
$$
\boldsymbol{f}^T \boldsymbol{L} \boldsymbol{f}=\lambda \boldsymbol{f}^T \boldsymbol{D} \boldsymbol{f}=\lambda
$$
图拉普拉斯矩阵特征值的大小刻画了图的连通性
另外，有时候还会加其他正则项：
$$
\min _{\boldsymbol{f}} \boldsymbol{f}^T \boldsymbol{L} \boldsymbol{f} \quad \text { s.t. } \quad \boldsymbol{f}^T \boldsymbol{D} \boldsymbol{f}=1, \boldsymbol{f}^T \boldsymbol{f}=C
$$
 ![[Pasted image 20230223214441.png]]
 两个约束的等值线的交点就是可行解域
 第一个特征值为0，没有意义，代表图上的所有顶点都是一类
 - 谱图理论
如果一个图有k个不相连的分量，那么图拉普拉斯矩阵是分块对角矩阵而且前k个特征值都是0，也就是说0特征值的重数代表图中连通分量的个数
![[Pasted image 20230223215222.png]]
- 谱聚类算法
	- 选择图拉普拉斯矩阵k个特征值对应的前k个特征向量去做Kmeans聚类（在n维度上代表每个图的顶点，在k维度上其实就是提取出前k个图拉普拉斯矩阵）
	- 如何选择k：
		- eigengap: $\Delta_k=\left|\lambda_k-\lambda_{k-1}\right|$
		- 衰减的特别快说明是低秩矩阵
- 降维技术
	- principal component analysis (PCA)
		- 从数据相关性的角度出发寻找主成分
		- 求解准则重构误差最小化： $\min _{v_1, v_2, \ldots, v_k} \sum_{i=1}^n\left\|x_i-\lambda_{i 1} v_1-\cdots-\lambda_{i k} v_k\right\|_2^2$， 和回归思想基本一样，都是尽可能的使得所有样本点到截面的距离是最小的，这个实际可以写做：
$$
\min _{\boldsymbol{V}^T \boldsymbol{V}=\boldsymbol{I}} \sum_{i=1}^n\left\|\boldsymbol{x}_i-\boldsymbol{V} \boldsymbol{V}^T \boldsymbol{x}_i\right\|_2^2

$$

$$
\boldsymbol{V}^T=\left(\begin{array}{c}
\boldsymbol{v}_1^T \\
\vdots \\
\boldsymbol{v}_k^T
\end{array}\right) \Longrightarrow \boldsymbol{V}^T \boldsymbol{x}_i=\left(\begin{array}{c}
\boldsymbol{v}_1^T \\
\vdots \\
\boldsymbol{v}_k^T
\end{array}\right) \boldsymbol{x}_i=\left(\begin{array}{c}
\lambda_{1 i} \\
\vdots \\
\lambda_{k i}
\end{array}\right) \Longrightarrow \lambda_{i 1} \boldsymbol{V}_1+\cdots+\lambda_{i k} \boldsymbol{x}_i=\boldsymbol{v}_k
$$
- 重构时可以使用L1距离，闵可夫斯基距离，此时会得到一个新的模型，它的解并不是特征值分解
$$
\begin{align}

\sum_{i=1}^n\left\|\boldsymbol{x}_i-\boldsymbol{V} \boldsymbol{V}^T \boldsymbol{x}_i\right\|_2^2 & =\sum_{i=1}^n\left(\boldsymbol{x}_i-\boldsymbol{V} \boldsymbol{V}^T \boldsymbol{x}_i\right)^T\left(\boldsymbol{x}_i-\boldsymbol{V} \boldsymbol{V}^T \boldsymbol{x}_i\right) \\ \\
& =\sum_{i=1}^n\left(\boldsymbol{x}_i^T \boldsymbol{x}_i-2 \boldsymbol{x}_i^T \boldsymbol{V} \boldsymbol{V}^T \boldsymbol{x}_i+\boldsymbol{x}_i^T \boldsymbol{V} \boldsymbol{V}^T \boldsymbol{V} \boldsymbol{V}^T \boldsymbol{x}_i\right) \\

& \Rightarrow \sum_{i=1}^n\left(-\boldsymbol{x}_i^T \boldsymbol{V} \boldsymbol{V}^T \boldsymbol{x}_i\right)=\sum_{i=1}^n \operatorname{tr}\left(-\boldsymbol{V}^T \boldsymbol{x}_i \boldsymbol{x}_i^T \boldsymbol{V}\right) \\
& =\operatorname{tr}\left(-\boldsymbol{V}^T \sum_{i=1}^n\left(\boldsymbol{x}_i \boldsymbol{x}_i^T\right) \boldsymbol{V}\right)=\operatorname{tr}\left(-\boldsymbol{V}^T \boldsymbol{X} \boldsymbol{X}^T \boldsymbol{V}\right)
\end{align}


$$
trace_norm是非常常见的优化形式，所有的二次型优化最终都会变成trace_norm，最终的优化问题变成了：
$$
\max _{\boldsymbol{V}^T \boldsymbol{V}=\boldsymbol{I}} \operatorname{tr}\left(\boldsymbol{V}^T \boldsymbol{X} \boldsymbol{X}^T \boldsymbol{V}\right)
$$
拉格朗日走一波：
$$
\mathcal{L}=\operatorname{tr}\left(\boldsymbol{V}^T \boldsymbol{X} \boldsymbol{X}^T \boldsymbol{V}\right)+\operatorname{tr}\left(\left(\boldsymbol{I}-\boldsymbol{V}^T \boldsymbol{V}\right) \boldsymbol{\Lambda}\right)
$$
其中$\boldsymbol{\Lambda}=\operatorname{diag}\left(\lambda_1, \ldots, \lambda_k\right) \neq 0$
$$
\frac{\partial \mathcal{L}}{\partial \boldsymbol{V}}=2 \boldsymbol{X} \boldsymbol{X}^T \boldsymbol{V}+\mathbf{2 V} \boldsymbol{\Lambda}=\mathbf{0} \Rightarrow \boldsymbol{X} \boldsymbol{X}^T \boldsymbol{V}=\boldsymbol{V} \boldsymbol{\Lambda}
$$
协方差矩阵n比较大d比较小的时候做特征值分解复杂度低，d比较大n比较小的时候做奇艺值分解
只要原问题有解， 那原问题的解必然是它的特征值分解的解， PCA的缺点缺乏可解释性
- 非负矩阵分解
![[Pasted image 20230302143906.png]]
 分成左矩阵和右矩阵发现右矩阵具有可解释性，也就是说pattern是通过两组pattern通过线性组合得到的，左边的矩阵称为字典具有特定的含义，而右边的矩阵coding就是权重矩阵来对字典进行线性组合。
对矩阵做pca以后会破坏数据中的含义，而非负矩阵分解不会
非负矩阵分解的定义是给定一个非负矩阵$\boldsymbol{X} \in \mathbb{R}_{+}^{d \times n}$, 分解成一个k列和k行的左矩阵和右矩阵$\boldsymbol{X} \approx L \boldsymbol{R}$, 目标函数是$\min _{\boldsymbol{L}, \boldsymbol{R}}\|\boldsymbol{X}-\boldsymbol{L} \boldsymbol{R}\|_F^2$, 其中$\boldsymbol{L} \in \mathbb{R}_{+}^{d \times k} \text { and } \boldsymbol{R} \in \mathbb{R}_{+}^{k \times n} \text { 都是非负矩阵 }$
非负矩阵分解的好处在于
- 只有加法，没有减法， 要求模型学到pattern，避免抵消(Cancellation)效应
- block coordinate descent(CD), 初始化对模型影响比较大
	- 选择初始点L0和R0
	- 如果不收敛，交替优化左矩阵和右矩阵
	- 投影梯度下降，先放弃约束找到最优解，然后投影到可行域中
- 交替式非负最小二乘 $\underset{\boldsymbol{L} \in \mathbb{R}_{+}^{d \times k}}{\operatorname{argmin}}\|\boldsymbol{X}-\boldsymbol{L} \boldsymbol{R}\|_F^2 \text { and } \underset{\boldsymbol{R} \in \mathbb{R}_{+}^{k \times n}}{\operatorname{argmin}}\|\boldsymbol{X}-\boldsymbol{L} \boldsymbol{R}\|_F^2$, 假设没有约束求伪逆作为初始解 $\boldsymbol{L}=\left[\boldsymbol{X} \boldsymbol{R}^{\dagger}\right]_\epsilon \text { and } \boldsymbol{R}=\left[\boldsymbol{L}^{\dagger} \boldsymbol{X}\right]_\epsilon$
- 改进设置自适应学习速率，可以将梯度下降简化为$\boldsymbol{L} \leftarrow \boldsymbol{L} \circ \frac{D R^T}{L R R^T} \text { and } \boldsymbol{R} \leftarrow \boldsymbol{R} \circ \frac{L^T X}{L^T L R}$, 乘法可以天然满足非负性，同时为了避免除0，需要在分母加一个Laplace系数$\epsilon$
降维方法的通项公式 $\min _{\boldsymbol{D} \in \mathcal{D}} \mathbb{E}_{\boldsymbol{x}}[L(\boldsymbol{x}, \boldsymbol{D})] \quad \text { where } \quad L(\boldsymbol{x}, \boldsymbol{D})=\min _{\boldsymbol{\alpha} \in \mathcal{A}} \frac{1}{2}\|\boldsymbol{x}-\boldsymbol{D} \boldsymbol{\alpha}\|^2+\lambda \phi(\boldsymbol{\alpha})$
- 稀疏编码
![[Pasted image 20230302153336.png]]



