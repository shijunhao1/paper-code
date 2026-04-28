# PPF-MR 实验代码

## 方法对应

代码严格按论文三模块组织：

1. 模式感知表示学习
- 局部上下文：带重启随机游走（RWR）
- 扩展上下文：普通随机游走（RW）
- 节点级一致性 + 结构级一致性（InfoNCE）联合优化

2. 传播感知候选模式生成
- 传播分数：节点度 + 邻居K-Core均值
- 选取高传播节点为种子初始化候选
- 综合归属分数：结构联系强度 + 全局关联程度 + 表示相似度
- 基于节点自适应阈值进行多归属分配

3. 候选精炼与目标匹配
- MLP 对候选内部节点打保留置信度
- BCE 监督训练
- 蒸馏后候选与目标样例做表示匹配

## 文件说明

- `run_ppfmr.py`：主入口
- `ppfmr.py`：核心算法实现
- `data_utils.py`：数据加载
- `model.py`：GNN编码器
- `metrics.py`：F1/Jaccard评价
- `utils.py`：工具函数

## 消融开关

- `--wo_node_consistency`：去掉节点级一致性约束
- `--wo_struct_consistency`：去掉结构级一致性约束
- `--wo_propagation_feature`：去掉传播特征（退化为度排序种子）
- `--wo_refinement`：去掉候选精炼模块

## 论文默认关键参数

- GCN, hidden=128, layers=2
- lr=0.001
- pretrain_batch_size=32
- pretrain_epoch=30
- walk_len=128, restart_prob=0.8
- temperature=0.5, lambda_struct=1.0
- alpha/beta/gamma=0.4/0.3/0.3
- refine_epoch=30, refine_threshold=0.5
- 候选数量：facebook=200, lj=1000, amazon/dblp/twitter=5000, cmnee-pr=2500
