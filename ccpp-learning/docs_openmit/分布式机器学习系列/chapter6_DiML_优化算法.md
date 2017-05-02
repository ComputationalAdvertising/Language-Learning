## 优化算法


### [FM＋SGD](http://blog.csdn.net/itplus/article/details/40536025)

### ffm模型＋（sgd／ftrl）

### ffm实现相关

用$n、m、k$分别表示域的个数、特征个数和隐向量长度.

+ 模型参数个数：$n \times m \times k$，稀疏模型：$n \times m_1 \times k. \;(m_1 \ll m )$

如何学习ffm模型的稀疏参数表示？

1. FFM模型的稀疏性如何得到？（涉及到w,v）
    1. 稀疏性：在admm 和 libffm 的实现中，并没有考虑稀疏性求解 ，model-ffm中考虑了稀疏性h_[n] = true/false;
    2. 线性项：admm中有体现，而在libffm和model-ffm 中未考虑线性项；
    3. 我们的设计中，可以不在ffm中添加线性项，线上用”pred(ffm+lr)” 实现 或者添加线性项，Entry需要设计更灵活；

    ```c++
    struct Entry {
        float w_;
    };
    
    struct FMEntry : public Entry {
        int num_field_;
        int k = 0;
        ps::SArray<float> sarray_.resize(num_field_ * k);   // 兼容ffm参数
    };
    ```

2. ffm在ps的worker端应该实现哪些逻辑 以及如何实现?

    ```c++
    // 第1步：预测。实现Predict函数. 
    // (确认pull操作对应全局model参数还是局部model参数？（如果是局部model参数会有问题）)
    wTx = Predict(const Row & row, Model model);  
    // 第2步：损失函数。累加损失函数(以交叉熵损失为例)
    expnyt = exp(-y * wTx);
    train_loss += log(1 + expnyt); // train_loss += log(1 + exp(-y * wTx))
    // 第3步：梯度计算。实现Gradient函数
    grad = Gradient(y, expnyt) = -y * expnyt / (1+expnyt);  // 针对每个j特征的隐向量？ 
    ```

交叉熵损失推导过程：

$$
\begin{align}   & \min_w \quad \frac{1}{m} \sum_{i=1}^{m} - \log (h_w(y^{(i)} x^{(i)})) \qquad\qquad(1) \\\   \Longrightarrow \; & \min_w \quad \frac{1}{m} \sum_{i=1}^{m} \underline{ \log (1+\exp(-y^{(i)} w^T x^{(i)})) } \quad\;\,(2)\\\   \Longrightarrow \; & \min_w \quad \underbrace { \frac{1}{m} \sum_{i=1}^{m} \underline{err(w,y^{(i)},x^{(i)})} }_{E_{in}(w)} \qquad\qquad\quad\, (3) \\\   \end{align} \qquad (ml.1.1.24)
$$

梯度计算公式：直接对w求偏导即可.

spacex demo Worker实现逻辑：

Predict接口：

```c++
virtual std::vector<ValueType> Predict(const RowBlock<IndexType>& row_block,                                        const std::vector<IndexType>& keys,                                        const std::vector<ValueType>& vals, 
                                        const std::vector<int>& lens) = 0;
// 1. 把worker pull下来的参数(vector格式)转化为map结构;
weight.insert(std::pair<IndexType, ValueType>(keys[i], vals[i]));
// 2. 从weight中取出参数，做累加计算（lr的逻辑）
weight.insert(std::pair<IndexType, ValueType>(keys[i], vals[i]));
```

> **⚠️：ffm的predict阶段要求keys必须是“全局参数”，因为涉及到二次交叉项特征的计算。**
> 应该不存在这样的问题，因为每一个worker会把所有featid对应的参数都拉下来。而这些参数已经覆盖了worker计算所需要的参数集合。




3. ffm在ps的server端该如何实现？(参数更新可否分布式环境实现)
    
    对于Server端输入参数结构：
    
    ```c++
    std::unordered_map<uint32_t, Entry> store_;
    ```
    
    问题：
    
    
    
    
    
参考model-ffm实现。 



