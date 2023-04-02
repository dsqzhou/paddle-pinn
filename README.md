# Paddle Hackathon 第4期 科学计算—— 科学计算方向 197
 
## 1.背景
原文：[Robust Regression with Highly Corrupted Data via PhysicsInformed Neural Networks](https://arxiv.org/abs/2210.10646)

参考：[robust_pinn](https://github.com/weipengOO98/robust_pinn)

- 近年来，大量文献研究表明PINN可以基于少量数据和PDE方程解决各类正向和反向问题。但是，当测量数据因仪器等原因误差较大时，求解精度无法保证。因为最小二乘法在数据偏差较大时会放大偏差，该文章提出用LAD-PINN(数据损失函数基于L1范数的PINN)来求解数据异常下的PDE和未知参数，并提出了基于中值绝对偏差的两阶段PINN方法(MAD-PINN)，以提高求解结果的准确性。
- 在MAD-PINN中，LAD-PINN作为异常数据检测器用于一次求解，然后通过中指绝对偏差筛选出偏差较大的数据，基于剩余正常数据，以L2范数为损失函数的PINN（OLS-PINN）对PDE进行重新求解。
- 在数据异常检测中，作者提供了两种数据清理准则：
    - 以最大误差为基准固定百分比，设置数据清理偏差范围
    - 假设数据误差服从正态分布，利用MAD估计标准差，并用其构造一个阈值来排除异常值：
    $\hat{\partial}_c=\frac{1}{1.6777}+\operatorname{median}\left(\left|u_i-\hat{a}^{+}\left(t_i, x_i\right)\right|\right)$
- 文章利用四个问题：泊松方程、波动方程、稳态和非稳态的NS方程进行方法验证。

## 2.代码说明
### 额外库需求
- paddle
- pyDOE

###  相关运行结果[Robust_PINNs4paddle AI studio](https://aistudio.baidu.com/aistudio/projectdetail/5864276)
  - 代码文件

    - poisson/* 论文Section 4.1. 泊松方程
    - piv/* 论文Section 4.2. 稳态二维圆柱绕流（层流）
    - wave/* 论文Section 4.3. 一维波动方程
    - ns/* 论文Section 4.4. 非稳态二维圆柱绕流
      - unsteady_NS.py为一阶段运行脚本
      - unsteady_NS_two_stage.py为二阶段运行脚本
      - run_size.sh和unsteady_NS_noise.py分别执行两类对比循环
    - basic_model.py 中为实现的全连接神经网络结构
    - parser_pinn.py 中包含了配置参数
    - gen_*_data.py中包含生成真实场及异常数据的函数
  - fig文件夹包含与论文对比图片，序号与文献一一对应

### 数据库
- [steady NS](https://github.com/dsqzhou/paddle-pinn/blob/main/piv/FluentSol.mat)
- [unsteady NS](https://github.com/dsqzhou/paddle-pinn/blob/main/ns/cylinder_nektar_wake.mat)

## 3.环境依赖
### 特别提示，Section4.4中用到了三阶微分，高阶自动微分需要在develop版本下静态图模式运行，安装以下版本
```
python -m pip install paddlepaddle-gpu==0.0.0.post112 -f https://www.paddlepaddle.org.cn/whl/linux/gpu/develop.html
export LD_LIBRARY_PATH=/opt/conda/envs/python35-paddle120-env/lib:${LD_LIBRARY_PATH}
```
- 同时，LBFGS优化器也需要在该版本下运行

## 计算结果
### 4.1 泊松方程
* 原始方程为1-D Poisson方程：
$$
u_{xx}=-16sin(4x) \in[-\pi, \pi]
$$
* 方程解析解为：
                 $$u(x)=sin(4x)+1$$
观测数据仅分布在$[-\pi, 0]$，需确定$[-\pi, \pi]$内的解
* Figure3-4证明了在无物理约束下，神经网络没有物理定律的泛化能力，无法依赖$[-\pi, 0]$的数据学习到$[0, \pi]$内的正确解。并且，针对噪声数据，神经网络在充足训练下会过度拟合数据。而加入PDE损失后，两种PINN求解结果都与实际基本吻合。
* Figure5显示了LAD与OLS再应对异常高值时的预测能力，在仅包含一个异常高值（x=0,y=10）的情况，OLS-PINN在$x \in[0, \pi]$的预测能力相比较差，曲线略微倾斜于异常点，说明L2范数会放大个别大误差的影响，导致整体预测的偏颇。与论文相比，复现结果中OLS的预测效果明显更好。
