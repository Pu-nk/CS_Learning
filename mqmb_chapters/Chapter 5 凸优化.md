# Introduction
## mathematical optimization
形如以下形式的都是数学优化问题：
$$
\begin{array}{ll}
\operatorname{minimize} & f_0(x) \\
\text { subject to } & f_i(x) \leq b_i, \quad i=1, \ldots, m
\end{array}
$$
对于所有的目标函数$f_0(x)$和约束函数$f_i(x)$,如果它们都是线性的，那么这个优化问题就被称为线性规划：
$$
f_i(\alpha x+\beta y)=\alpha f_i(x)+\beta f_i(y)
$$
同样，如果优化问题是非线性的，那么就称为非线性优化问题。而所谓凸优化是指目标函数和约束函数都是凸函数，即：
$$
f_i(\alpha x+\beta y) \leq \alpha f_i(x)+\beta f_i(y)
$$
其中$\alpha$和$\beta$都在$0-1$之间，且$\alpha+\beta = 1$

# Convex sets
- Affine set
	- **line** through $x_{1}$, $x_{2}$: all points
		- $x = \theta x_{1} + (1 - \theta)x_{2} \ \ x\in R$  
	- **affline set** contains the lines through any two distinct points in the set
	- 任何一个仿射集可以作为一个线性方程组的解集
- convex set
	- 包含集合中任意两点所连接的线段的区域
	- $x_{1}, x_{2} \in \mathbb{C}, 0 \leq \theta \leq 1 \rightarrow \theta x_{1} +(1-\theta)x_{2} \in \mathbb{C}$
- convex combination and convex hull
	- **convex combination** of $x_{1}, x_{2},\cdots x_{k}$: any point $x$ of the form
		- $x = \theta_{1}x_{1} + \theta_{2}x_{2} + \cdots +\theta_{k}x_{k}$
		- $\sum_{i=1}^{k}\theta_{i} = 1$
	- **convex hull**: set of all convex combinations of points in $S$
- convex cone (凸锥)
	- conic combinations: $x = \theta x_{1} +(1 - \theta)x_{2}$ 
		- $\theta \geq 0  \ \text{and}\  (1-\theta)\geq 0$
- Hyperplanes and halfspaces
	- hyperplane: set of the form $\{x | a^{T}x = b\}(a \neq 0)$
	- halfspace: set of the form $\{x | a^{T}x \leq b\}(a \neq 0)$
- Euclidean ball and ellipsoids
	- ball with center $x_{c}$ and radius $r$:
		- $B(x_{c}, r) = \{x | \|x - x_{c}\|_{2} \leq r\} = \{x_{c} + ru|\|u\|_{2} \leq 1\}$
	- ellipsoid: set of the form
		- $\{x | (x - x_{c})^{T}P^{-1}(x - x_{c}) \leq 1\} = \{x_{c}+Au |\|u\|_{2} \leq 1\}$ 
		- $P \in S^{n}_{++}$ 正定矩阵
```ad-note
椭球是欧几里得球的推广
```
		
- norm ball and norm cones
- Polyhedra (多面体)
	- 有限多个等式和不等式的解集
	- $Ax \preceq b$, $Cx = d$(该向量的每一个元素都小于b)
	- 多面体就是不等式方程组的每一条边围成的
- Operations that preserve convexity
	- intersection：任意个凸集的交集还是凸集
	- Affine function：仿射函数保留正向和反向凸度
	- perspective and line-fractional function
		- perspective function: $P(x,t) = \frac{x}{t}, \text{dom} \ P = \{(x, t)| t > 0\}$, (向量的前n个元素除以最后一个元素$\mathbb{R}^{n+1}\rightarrow \mathbb{R}^n$)images and inverse images of convex sets under perspective are convex
		- line-fractional function: $f(x) = \frac{Ax + b}{c^T x + d}, \text{dom}\ f=\{x|c^tx+d>0\}$, images and inverse images of convex sets under line-fractional are convex
		- 