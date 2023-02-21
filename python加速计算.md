## loop

```python
import random
def average_py(n):
	s = 0
	for i in range(n):
		s += random.random()
	return s / n
# numpy 加速
import numpy as np
def average_py(n):
	s = np.random.random(n)
	return s.mean()
# numba 加速
import numba
average_nb = numba.jit(average_py)
average_nb(n)
# cpython
```
## algorithm
```python
def is_prime_py(I):
	if I % 2 == 0: return False
	for i in range(3, int(I * 0.5)+1, 2):
		if I % i ==0: return False
	return True

# numba 加速
is_prime_nb = numba.jit(is_prime_py)

# cpython

# multiprocessing
import multiprocessing as mp
pool = mp.Pool(processes=4)
pool.map(is_prime, input)
```

## summary
提升代码计算速度的一些方法汇总如下：
- 高效的python范式：如向量化、列表表达式
- 使用一些数据框架：如numpy、pandas 定义了很多底层是C驱动的计算函数
- 优化代码编译速度：如numba、Cpython（支持静态编译）
- 并行化：如joblib、multiprocessing
- 