# 六个路由对照

C1单组4logits；C2两组4logits按[short,short,long,long]；C3四组4logits。C4 context=[LN(s),LN(t),LN(Zlast-mean5Z),LN(Linear128→64(mean20H))]，256→32→4，GELU。C5/C6在context后拼接4×8周期embedding，264→32→4。温度1，四专家全部可访问。最后层Normal(0,.001)、bias0；embedding Normal(0,.02)。C4与C5的共同context变换、第一层context列及最后层显式复制，C5/C6全参数相同；容量差288参数记录为限制。

静态路由不读取样本，动态路由不读取标签，无固定专家掩码、warmup、entropy或balance loss。C5 head Dropout作用于expert，C6作用于horizon：同shape/随机数数量但作用对象不同，不能宣称训练轨迹相同。
