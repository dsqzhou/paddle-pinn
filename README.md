# paddle-pinn
 
- 复现论文：Robust Regression from Highly Corrupted Data via Physics-informed Neural Network

## 额外库需求
- paddle
- pyDOE

## 代码文件

- poisson/* 论文Section 4.1. 泊松方程
- piv/* 论文Section 4.2. 稳态二维圆柱绕流（层流）
- wave/* 论文Section 4.3. 一维波浪方程
- ns/* 论文Section 4.4. 非稳态二维圆柱绕流

## 数据库
- [steady NS](https://github.com/dsqzhou/paddle-pinn/blob/main/piv/FluentSol.mat)
- [unsteady NS](https://github.com/dsqzhou/paddle-pinn/blob/main/ns/cylinder_nektar_wake.mat)
