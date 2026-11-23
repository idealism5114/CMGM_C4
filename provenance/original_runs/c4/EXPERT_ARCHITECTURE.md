# 共享四专家

E1=ReLU(W[Z_last,mean5Z,Z_last-mean5Z])，12352参数。它复用旧RecentChange的投影机制并显式增加均值输入，比旧128→64增加4096参数。Z包含递推历史，不是仅最近五日原始数据。

E2=ReLU(W[H_last,mean20H])，16448参数，直接复用LongMemoryExpert。

E3=s+ReLU(Ws)，4160参数，直接复用SpatialStructureExpert。

E4=Wo[(Us s)⊙(Ut t)]，rank16，3072参数；复用旧LowRankInteraction矩阵，按本轮公式去掉旧t残差。全部MoE共享同一专家定义及初值。

四专家每次各计算一次。Prediction MoE：一次共享Head body作用于B,K,64，然后输出权重view(H,C,64)取对应行，得到B,H,K,C，再乘路由。C6只改变融合位置；初始state_dict与C5完全相同。没有复制专家头、输出归一化或辅助损失。
