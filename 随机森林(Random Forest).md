随机森林(Random Forest)是一种由诸多决策树通过 Bagging 的方式组成的分类器。如图表 3 所示，我们由原始数据集生成 N 个 Bootstrap 数据集，对于每个 Bootstrap 数据集分别训 练一个弱分类器，最终用投票、取平均值等方法组合成强分类器。
![[Pasted image 20221110203824.png]]