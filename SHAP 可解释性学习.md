## 背景
模型可解释性是机器学习在金融领域应用的重大挑战
## 方法
- 特征重要性 Feature Importance
	- 特征重要性是根据某个特征在决策树分裂时前后的信息增益
	- 单步的特征重要性是指分裂前后 Gini指数增益的和，而对于单棵决策树特征 A的重要性定义为所有按特征 A进行分裂的节点，
- ICE (Individual Conditional Expectation)和PDP(Partial Dependence Plot)
	- 基本上就是灵敏度分析，考察某项特征的不同取值对模型输出值的影响
	- 前者侧重单一样本，后者侧重多条样本
- SDT(Surrogate Decision Trees)
	- 代理决策树是用单棵决策树解释其它更复杂的机器学习模型
	- 缺点在于当模型高度复杂时，简单的决策树可能不足以刻画决策规则，并且用单棵决策树拟合原模型可能引入新的误差。
	- 但是模型的归纳偏置不同，预测的结果也不同，如何说明这两者是同构的？（只有预测结果充分接近的前提下）
			![[Pasted image 20221111111435.png]]
- LIME(Local Interpretable Model-agnostic Explanations)
	- 选一条样本x，并在x邻域内扩充样本训练一个可解释的代理模型根据回归系数来解释样本。
	- 具体代码实现见[GitHub - marcotcr/lime: Lime: Explaining the predictions of any machine learning classifier](https://github.com/marcotcr/lime)
![[Pasted image 20221111130656.png]]
- SHAP
	- 计算特征对模型输出的边际贡献
	- 对于不同的特征排列计算不同的特征X的边际贡献，最后根据组合数作为权重求平均
	- 具体代码实现见 [GitHub - slundberg/shap: A game theoretic approach to explain the output of any machine learning model.](https://github.com/slundberg/shap)
![[Pasted image 20221111132420.png]]
![[Pasted image 20221111102729.png]]