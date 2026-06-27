# LLM & Agent 应用及后训练面试准备指南

> 基于 Megatron-LM 框架的深度技术面试准备文档
> 适用于：LLM算法实习、Agent应用、后训练(Post-Training)方向

---

## 目录

1. [分布式训练核心理论](#1-分布式训练核心理论)
2. [Megatron-LM 并行策略深度解析](#2-megatron-lm-并行策略深度解析)
3. [MoE (Mixture of Experts) 技术栈](#3-moe-mixture-of-experts-技术栈)
4. [后训练与RLHF技术](#4-后训练与rlhf技术)
5. [内存优化与性能调优](#5-内存优化与性能调优)
6. [Agent框架与应用](#6-agent框架与应用)
7. [前沿模型架构](#7-前沿模型架构)
8. [面试深挖问题与参考答案](#8-面试深挖问题与参考答案)
9. [项目经验包装建议](#9-项目经验包装建议)

---

## 1. 分布式训练核心理论

### 1.1 数据并行 (Data Parallelism, DP)

**核心思想**：每个GPU持有完整模型副本，数据被分割到不同GPU上。

**关键概念**：
- **AllReduce**：梯度同步的核心操作，所有GPU的梯度求平均
- **Ring-AllReduce**：NVIDIA NCCL使用的高效通信算法，通信量为 `2(N-1)/N * M`（N为GPU数，M为参数量）
- **梯度累积**：在不增加通信的情况下模拟大batch size

**Megatron实现**：
```python
# megatron/core/distributed/ 目录
# DDP实现：梯度bucket化 + 异步通信
# FSDP2集成：PyTorch原生FSDP支持
```

**面试考察点**：
1. DP vs DDP的区别？为什么DDP更快？
2. 如何计算AllReduce的通信量？
3. 什么时候梯度累积比增大batch size更好？

### 1.2 模型并行 (Model Parallelism)

#### 1.2.1 张量并行 (Tensor Parallelism, TP)

**核心思想**：将单个层的参数矩阵分割到多个GPU上。

**Megatron实现** (`megatron/core/tensor_parallel/layers.py`)：

```python
class ColumnParallelLinear(nn.Module):
    """
    Y = XA，A按列分割：A = [A_1, A_2, ..., A_n]
    每个GPU计算：Y_i = X * A_i
    """
    
class RowParallelLinear(nn.Module):
    """
    Y = XA，A按行分割：A = [A_1; A_2; ...; A_n]
    每个GPU计算：Y_i = X_i * A_i
    最终结果：Y = AllReduce(Y_1 + Y_2 + ... + Y_n)
    """
```

**TP通信模式**：
- **Column Parallel**：输入AllGather/Scatter，输出无需通信（或ReduceScatter用于Sequence Parallel）
- **Row Parallel**：输入无需通信，输出AllReduce

**序列并行 (Sequence Parallelism)**：
- 将LayerNorm、Dropout等非并行操作沿序列维度分片
- 配合TP使用：ReduceScatter → 各GPU独立计算 → AllGather
- 节省显存：激活内存从O(batch * seq * hidden)降到O(batch * seq/TP * hidden)

**面试考察点**：
1. 画出Column Parallel和Row Parallel的计算图
2. 为什么MLP层天然适合TP？Attention层呢？
3. Sequence Parallel如何减少显存？通信开销如何？

#### 1.2.2 流水线并行 (Pipeline Parallelism, PP)

**核心思想**：将模型的不同层分配到不同GPU上，形成流水线。

**Megatron实现** (`megatron/core/pipeline_parallel/schedules.py`)：

**经典1F1B调度**：
```
GPU0: F0 F1 F2 F3 B3 B2 B1 B0
GPU1:    F0 F1 F2 F3 B3 B2 B1 B0
GPU2:       F0 F1 F2 F3 B3 B2 B1 B0
GPU3:          F0 F1 F2 F3 B3 B2 B1 B0
```

**虚拟流水线并行 (Virtual Pipeline Parallelism, VPP)**：
- 将层分成更多chunk，交错分配到各stage
- 减少pipeline bubble：bubble率从 `(p-1)/(m+p-1)` 降到更小
- Megatron支持非对称VPP：`--pipeline-model-parallel-layout`自定义布局

**关键参数**：
- `micro_batch_size`：微批次大小
- `num_microbatches`：微批次数
- `pipeline_model_parallel_size`：PP并行度

**面试考察点**：
1. 计算1F1B的bubble率和吞吐量
2. VPP如何减少bubble？代价是什么？
3. PP和TP如何组合使用？

#### 1.2.3 专家并行 (Expert Parallelism, EP)

**核心思想**：MoE模型中，不同专家分配到不同GPU。

**Megatron创新 - MoE Parallel Folding**：
```
传统：EP ≤ DP，专家数量受限于DP度
创新：解耦attention和MoE层的并行策略
- Attention层：TP × CP × DP × PP
- MoE层：ETP × EP × EDP × PP
```

**DeepEP和HybridEP后端**：
- 支持TMA (Tensor Memory Accelerator) + IBGDA的高性能token dispatch
- 优化AllToAll通信

**面试考察点**：
1. EP和DP的关系？为什么需要MoE Parallel Folding？
2. AllToAll通信在MoE中的作用？
3. 如何处理专家负载不均衡？

### 1.3 上下文并行 (Context Parallelism, CP)

**核心思想**：将长序列沿context维度分片，每个GPU处理序列的一部分。

**Dynamic Context Parallelism**：
- 自适应CP大小，支持可变长序列
- 训练加速1.48x

**面试考察点**：
1. CP和Sequence Parallel的区别？
2. 长序列训练的挑战有哪些？

---

## 2. Megatron-LM 并行策略深度解析

### 2.1 5D并行策略

Megatron-LM支持的5维并行：

| 并行维度 | 符号 | 作用 | 通信操作 |
|---------|------|------|---------|
| 张量并行 | TP | 分割层内参数 | AllReduce/ReduceScatter |
| 流水线并行 | PP | 分割模型层 | P2P Send/Recv |
| 数据并行 | DP | 分割数据 | AllReduce |
| 专家并行 | EP | 分割MoE专家 | AllToAll |
| 上下文并行 | CP | 分割序列 | AllGather/ReduceScatter |

### 2.2 并行策略选择指南

**小模型 (< 10B)**：
- TP=2/4 (单节点内)
- DP=剩余GPU数
- PP=1

**大模型 (> 100B)**：
- TP=4/8 (节点内)
- PP=2/4/8 (跨节点)
- DP=剩余GPU数

**MoE模型**：
- EP=专家数 (或其因数)
- TP=1/2 (用于attention)
- EDP=DP/EP

**面试考察点**：
1. 为什么TP通常在节点内，PP跨节点？
2. 如何根据模型大小选择并行策略？
3. 通信瓶颈分析：哪个并行维度最容易成为瓶颈？

### 2.3 通信优化

**梯度通信重叠**：
```bash
--overlap-grad-reduce      # 梯度reduce与后向计算重叠
--overlap-param-gather     # 参数all-gather与前向计算重叠
```

**TP通信重叠**：
```bash
--tp-comm-overlap          # TP通信与计算重叠
```

**PP通信优化**：
- 异步P2P通信
- 通信与计算重叠

**面试考察点**：
1. 梯度通信重叠的实现原理？
2. 如何判断通信是否成为瓶颈？

---

## 3. MoE (Mixture of Experts) 技术栈

### 3.1 MoE核心组件

**Router** (`megatron/core/transformer/moe/router.py`)：

```python
class TopKRouter(Router):
    """
    路由流程：
    1. 计算logits：g(x) = W_gate * x
    2. 选择top-k专家
    3. 计算路由权重（softmax/sigmoid）
    4. 可选：应用token dropping
    """
```

**负载均衡策略**：
- `aux_loss`：Switch Transformer的辅助损失
- `seq_aux_loss`：序列级辅助损失
- `global_aux_loss`：全局辅助损失
- `sinkhorn`：Sinkhorn分配
- `aux-loss-free`：DeepSeek-V2的动态偏置方法

**Token Dispatcher**：
- `AllToAll`：传统MoE通信
- `DeepEP`：高性能token dispatch
- `HybridEP`：混合dispatch策略
- `AllGather`：另一种dispatch方式

### 3.2 MoE训练挑战

**专家坍塌 (Expert Collapse)**：
- 现象：少数专家处理大部分token
- 解决：负载均衡损失、路由策略改进

**通信开销**：
- AllToAll通信量：`O(batch * seq * hidden * top_k)`
- 解决：EP优化、通信重叠

**显存占用**：
- 专家参数：`num_experts * expert_size`
- 解决：专家分片、CPU offloading

**面试考察点**：
1. MoE相比Dense模型的优缺点？
2. 如何解决专家坍塌问题？
3. DeepSeek-V2/V3的MoE创新点？

### 3.3 MoE Upcycling

**概念**：将预训练的Dense模型转换为MoE模型。

**Megatron实现**：
- 运行时将FFN层复制多份作为专家
- 添加Router层
- 继续训练

**面试考察点**：
1. MoE Upcycling的优势？
2. 如何初始化Router？

---

## 4. 后训练与RLHF技术

### 4.1 后训练概述

**后训练流程**：
```
预训练 → SFT → RLHF/DPO → 量化/蒸馏 → 部署
```

**Megatron后训练模块** (`megatron/post_training/`)：
- ModelOpt集成：量化、蒸馏、剪枝
- 知识蒸馏：logits缓存和蒸馏loss

### 4.2 GRPO (Group Relative Policy Optimization)

**核心算法** (`megatron/rl/rl_utils.py`)：

```python
def calculate_grpo_loss(
    current_logprobs,  # π(a|s) - 当前策略
    old_logprobs,      # π_old(a|s) - 采样策略
    ref_logprobs,      # π_ref(a|s) - 参考策略
    advantages,        # A(s,a) - 优势函数
    ...
):
    # 1. 计算重要性采样比率
    ratios = (current_logprobs - old_logprobs).exp()
    
    # 2. PPO裁剪
    clamped_ratios = ratios.clamp(1 - ε, 1 + ε)
    
    # 3. KL散度约束
    kl_term = (ref_logprobs - current_logprobs).exp() - (ref_logprobs - current_logprobs) - 1
    
    # 4. 熵正则化
    entropy_term = -current_logprobs.exp() * current_logprobs
    
    # 5. 最终损失
    loss = -min(ratios * A, clamped_ratios * A) + β * kl_term - λ * entropy_term
```

**GRPO vs PPO**：
- GRPO使用组内相对优势，无需额外的Critic网络
- 计算更简单，内存占用更小

**面试考察点**：
1. GRPO和PPO的区别？为什么选择GRPO？
2. 重要性采样比率的作用？
3. KL散度约束如何防止策略退化？
4. 熵正则化的作用？

### 4.3 Megatron-RL 架构

**核心组件**：

```python
# Agent抽象
class Agent:
    def get_grouped_rollouts(self, request: GroupedRolloutRequest) -> GroupedRollouts:
        """生成一组rollout"""
        pass

# InferenceInterface
class InferenceInterface:
    def generate(self, prompt, **generation_args):
        """生成文本"""
        pass

# Trainer/Evaluator
class Trainer:
    def train_step(self, rollout_data):
        """训练步骤"""
        pass
```

**训练-推理解耦**：
1. 训练模型和推理模型可以是同一个，也可以是分离的
2. 权重Refit：`megatron/core/resharding/refit.py`
3. 推理时offload优化器状态，训练时prefetch回GPU

**序列打包 (Sequence Packing)**：
- 将多个不等长序列打包到一个bin
- 使用`PackedSeqParams`处理变长序列
- 提高GPU利用率

**面试考察点**：
1. RLHF训练的主要挑战？
2. 为什么需要训练-推理解耦？
3. 序列打包如何提高效率？如何处理attention mask？

### 4.4 权重Refit

**问题**：RL训练中，训练模型和推理模型需要同步权重。

**Megatron方案**：
```python
# megatron/core/resharding/refit.py
def swap_model_weights(train_model, inference_model, method):
    """
    方法1: 直接拷贝（简单但慢）
    方法2: 分布式resharding（高效）
    """
```

**面试考察点**：
1. 权重Refit的时机？
2. 如何保证训练和推理模型的一致性？

---

## 5. 内存优化与性能调优

### 5.1 显存分析

**模型参数显存**：
```
FP32: params * 4 bytes
FP16/BF16: params * 2 bytes
FP8: params * 1 byte
```

**优化器状态显存**（Adam）：
```
FP32: params * 4 * 2 = params * 8 bytes (momentum + variance)
混合精度: params * 4 * 2 + params * 2 = params * 10 bytes
```

**激活显存**：
```
O(batch * seq * hidden * num_layers * bytes_per_element)
```

### 5.2 激活重计算 (Activation Checkpointing)

**Megatron实现** (`megatron/core/recompute.py`)：

**选择性重计算**：
- 只重计算部分层的激活
- 支持模块级粒度：`mla_up_proj`、`layernorm`、`moe_act`、`core_attn`、`mlp`、`moe`

**重计算策略**：
```python
# 完全重计算：所有层
--recompute-granularity full

# 选择性重计算：只重计算attention
--recompute-granularity selective
```

**面试考察点**：
1. 重计算的时间-空间tradeoff？
2. 选择性重计算如何选择哪些层重计算？

### 5.3 混合精度训练

**精度支持**：
- FP16：传统混合精度
- BF16：更好的数值稳定性
- FP8：E4M3/E5M2，进一步节省显存
- FP4：MXFP4，最新支持

**FP8训练** (`megatron/core/fp8_utils.py`)：

**FP8配方**：
- `per-tensor`：整个tensor共享scale
- `blockwise`：1x128激活 + 128x128权重
- `MXFP8`：Blackwell原生支持

**FP8 Primary Weights**：
- 直接从FP32 cast到FP8
- 消除BF16中间拷贝
- 节省显存和带宽

**面试考察点**：
1. BF16 vs FP16的区别？
2. FP8训练的挑战？如何防止数值溢出？
3. 梯度缩放(Gradient Scaling)的作用？

### 5.4 分布式优化器

**ZeRO-1实现**：
- 优化器状态分片到各DP rank
- 每个rank只存储1/DP的优化器状态
- 通信：AllGather获取完整参数，ReduceScatter分发梯度

**CPU Offloading**：
```bash
--optimizer-cpu-offload  # 优化器状态offload到CPU
```

**面试考察点**：
1. ZeRO-1/2/3的区别？
2. CPU Offloading的性能影响？

### 5.5 CUDA Graph

**Megatron支持**：
- per-layer CUDA Graph
- full-iteration CUDA Graph
- 动态batch size支持

**面试考察点**：
1. CUDA Graph的原理？
2. 为什么CUDA Graph能提高性能？限制是什么？

---

## 6. Agent框架与应用

### 6.1 Megatron-RL Agent架构

**Agent抽象** (`megatron/rl/agent/api.py`)：

```python
class Rollout:
    """单次rollout数据"""
    tokens: List[int]
    reward: float
    logprobs: List[float]
    
class RolloutGroup:
    """一组rollout，来自同一个prompt"""
    rollouts: List[Rollout]
    
class GroupedRolloutRequest:
    """rollout请求"""
    num_groups: int
    rollouts_per_group: int
    inference_interface: InferenceInterface
    generation_args: Dict
```

**Agent类型**：
- `WeightedMultiTask`：多任务加权
- `RemoteAgent`：远程Agent
- `RewardOnlyAgent`：纯奖励评估
- `PassAtEvaluationAgent`：pass@k评估

### 6.2 环境设计

**示例环境** (`examples/rl/environments/`)：
- Math环境：数学推理任务
- Countdown环境：倒计数游戏

**环境接口**：
```python
class Environment:
    def get_reward(self, response: str, ground_truth: str) -> float:
        """计算奖励"""
        pass
```

### 6.3 推理-训练协调

**MegatronLocal推理**：
- 支持CUDA Graph推理
- 动态批处理
- 对称内存管理

**分布式推理**：
- 推理服务器：`InferenceInterfaceServer`
- 支持HuggingFace/OpenAI后端

**面试考察点**：
1. Agent设计的关键考量？
2. 如何平衡推理延迟和训练吞吐？
3. 多任务Agent的挑战？

---

## 7. 前沿模型架构

### 7.1 Multi-Latent Attention (MLA)

**DeepSeek-V2/V3创新** (`megatron/core/transformer/multi_latent_attention.py`)：

**核心思想**：
- 压缩KV cache：将KV投影到低维空间
- 解耦RoPE：分离位置编码和内容编码

**实现**：
```python
class MultiLatentAttention(Attention):
    """
    KV压缩：
    1. 下投影：c_kv = W_dkv * x  (hidden → latent_dim)
    2. 上投影：k = W_uk * c_kv, v = W_uv * c_kv
    
    Q压缩：
    1. 下投影：c_q = W_dq * x
    2. 上投影：q = W_uq * c_q
    """
```

**KV Cache优化**：
- 传统：存储所有层的K和V
- MLA：只存储压缩的c_kv
- 节省显存：`latent_dim / (head_dim * num_heads)`

**面试考察点**：
1. MLA相比MHA/GQA的优势？
2. 如何实现位置编码的解耦？
3. MLA对KV cache的影响？

### 7.2 Multi-Token Prediction (MTP)

**DeepSeek-V3创新** (`megatron/core/transformer/multi_token_prediction.py`)：

**核心思想**：
- 一次预测多个未来token
- 使用多个预测头并行预测

**实现**：
```python
class MultiTokenPrediction(MegatronModule):
    """
    输入：hidden_states
    输出：多个token的预测概率
    
    每个预测头：
    1. 嵌入已预测token
    2. 与hidden_states结合
    3. 预测下一个token
    """
```

**面试考察点**：
1. MTP如何提高训练效率？
2. MTP对推理的影响？

### 7.3 Hybrid Transformer-Mamba

**架构创新** (`megatron/core/models/hybrid/`)：

**核心思想**：
- Transformer层：擅长长距离依赖
- Mamba层：线性复杂度，高效处理长序列
- 交替使用：平衡性能和效率

**Megatron支持**：
- Falcon-H1架构
- 自定义层分配策略

**面试考察点**：
1. Transformer和Mamba的优缺点？
2. 如何决定哪些层用Transformer，哪些用Mamba？

---

## 8. 面试深挖问题与参考答案

### 8.1 基础概念类

**Q1: 解释Megatron-LM的Tensor Parallelism是如何工作的？**

**参考答案**：
Tensor Parallelism将单个层的参数矩阵分割到多个GPU上。对于线性层Y = XA：
- **Column Parallel**：A按列分割为[A1, A2, ..., An]，每个GPU计算Yi = X * Ai，无需通信（如果后面接Row Parallel）
- **Row Parallel**：A按行分割为[A1; A2; ...; An]，每个GPU计算Yi = Xi * Ai，结果需要AllReduce

Megatron的设计模式是Column Parallel → Row Parallel成对使用，中间不需要通信，只在最后做一次AllReduce。

**Q2: 为什么MoE模型需要Expert Parallelism？**

**参考答案**：
MoE模型有大量专家参数（如DeepSeek-V3有256个专家），单个GPU无法存储所有专家。EP将专家分布到不同GPU上，通过AllToAll通信将token发送到对应的专家GPU。挑战是AllToAll通信开销大，Megatron通过DeepEP/HybridEP优化，以及MoE Parallel Folding解耦attention和MoE的并行策略来解决。

**Q3: GRPO相比PPO有什么优势？**

**参考答案**：
GRPO (Group Relative Policy Optimization) 使用组内相对优势，不需要额外的Critic网络：
1. **节省内存**：无需存储和更新Critic模型
2. **计算简单**：优势计算基于组内排序，而非GAE
3. **稳定性**：相对优势减少了reward scale的影响

Megatron-RL的实现还支持重要性采样修正、KL散度约束、熵正则化等。

### 8.2 实现细节类

**Q4: 序列打包(Sequence Packing)如何工作？**

**参考答案**：
序列打包将多个不等长序列打包到一个bin中，使用`PackedSeqParams`记录每个序列的位置：
```python
cu_seqlens = [0, 5, 12, 20]  # 累积序列长度
```
Attention层使用`cu_seqlens`来正确计算attention，避免跨序列attend。好处是减少了padding浪费，提高了GPU利用率。

**Q5: 如何实现训练-推理解耦？**

**参考答案**：
Megatron-RL支持两种模式：
1. **共享模型**：训练和推理使用同一个模型，推理时切换到eval模式
2. **分离模型**：训练模型和推理模型分开，通过权重Refit同步

分离模型的优势是可以优化推理（如CUDA Graph），但需要额外的显存和同步开销。Megatron支持UVM和torch_memory_saver来offload推理模型权重。

**Q6: FP8训练的挑战是什么？**

**参考答案**：
FP8的数值范围小（E4M3: -448到448），容易溢出。Megatron的解决方案：
1. **Per-tensor Scaling**：使用scale因子将数值映射到FP8范围
2. **Blockwise Scaling**：更细粒度的scale，适应不同数值分布
3. **延迟缩放(Deferred Scaling)**：使用前几步的最大值来更新scale
4. **主权重保持FP32**：避免累积误差

### 8.3 系统设计类

**Q7: 如何设计一个支持万亿参数模型训练的系统？**

**参考答案**：
需要组合多种并行策略：
1. **TP=8**：节点内，利用NVLink高带宽
2. **PP=8**：跨节点，使用P2P通信
3. **DP=64**：数据并行，使用AllReduce
4. **CP=4**：如果序列很长，使用Context Parallel

还需要：
- 分布式优化器（ZeRO-1/2/3）
- 激活重计算/Offloading
- 混合精度训练
- 高效的数据加载
- 检查点和容错

**Q8: 如何优化RLHF训练的性能？**

**参考答案**：
1. **训练-推理解耦**：推理时offload优化器，训练时prefetch回GPU
2. **序列打包**：减少padding浪费
3. **CUDA Graph**：减少kernel launch开销
4. **梯度累积**：减少通信频率
5. **异步rollout生成**：与训练重叠
6. **权重Refit优化**：使用分布式resharding

### 8.4 问题排查类

**Q9: 训练loss突然飙升，如何排查？**

**参考答案**：
可能原因：
1. **学习率过大**：检查lr schedule
2. **梯度爆炸**：检查梯度norm，使用gradient clipping
3. **数据问题**：检查数据质量，是否有异常样本
4. **数值溢出**：检查是否有NaN/Inf，使用FP8时检查scale
5. **通信问题**：检查是否有straggler

Megatron的调试工具：
- `--check-for-nan-in-loss-and-grad`：检查NaN
- `--check-for-spiky-loss`：检查loss spike
- Straggler检测：检测慢节点

**Q10: 如何分析和优化训练吞吐量？**

**参考答案**：
1. **MFU (Model FLOP Utilization)**：实际FLOPS / 理论FLOPS
2. **通信分析**：使用NCCL日志或profiler
3. **计算分析**：使用NVTX标记
4. **内存分析**：使用torch.cuda.memory_summary

优化方向：
- 增大batch size（减少通信比例）
- 使用梯度累积
- 优化通信重叠
- 使用CUDA Graph
- 调整并行策略

---

## 9. 项目经验包装建议

### 9.1 如果你使用过Megatron-LM

**可以强调的点**：
1. 熟悉大规模分布式训练架构
2. 理解5D并行策略及其tradeoff
3. 有MoE/RLHF训练经验
4. 了解内存优化技术
5. 有性能调优经验

**项目描述示例**：
> "使用Megatron-LM框架进行大模型分布式训练，深入理解Tensor Parallelism、Pipeline Parallelism等并行策略。通过调整并行度配置和优化通信重叠，将训练吞吐量提升了X%。"

### 9.2 如果你没有直接使用过Megatron-LM

**可以强调的点**：
1. 阅读过Megatron-LM源码，理解核心设计
2. 了解分布式训练的基本原理
3. 对比过不同框架（DeepSpeed、FSDP等）
4. 有相关领域的研究经验

**项目描述示例**：
> "深入研究了Megatron-LM源码，分析了其Tensor Parallelism和Pipeline Parallelism的实现细节。对比了Megatron-LM与DeepSpeed的MoE实现，总结了各自的优缺点。"

### 9.3 面试准备清单

**必须掌握**：
- [ ] Tensor Parallelism的Column/Row Parallel
- [ ] Pipeline Parallelism的1F1B调度
- [ ] MoE的基本原理和挑战
- [ ] GRPO/PPO算法
- [ ] 混合精度训练
- [ ] 激活重计算

**加分项**：
- [ ] MLA/MTP等前沿架构
- [ ] 序列打包实现
- [ ] 训练-推理解耦
- [ ] FP8训练细节
- [ ] CUDA Graph优化
- [ ] 实际调优经验

---

## 附录：关键代码位置

| 模块 | 文件路径 |
|------|---------|
| Tensor Parallelism | `megatron/core/tensor_parallel/layers.py` |
| Pipeline Parallelism | `megatron/core/pipeline_parallel/schedules.py` |
| MoE Router | `megatron/core/transformer/moe/router.py` |
| MLA | `megatron/core/transformer/multi_latent_attention.py` |
| GRPO Loss | `megatron/rl/rl_utils.py` |
| 训练循环 | `megatron/training/training.py` |
| RL训练入口 | `train_rl.py` |
| 分布式优化器 | `megatron/core/optimizer/` |
| FP8支持 | `megatron/core/fp8_utils.py` |
| 序列打包 | `megatron/rl/sequence_packing_utils.py` |
| 权重Refit | `megatron/core/resharding/refit.py` |

---

## 参考资源

1. **Megatron-LM论文**：
   - Shoeybi et al., "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism"
   - Narayanan et al., "Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM"

2. **MoE论文**：
   - Fedus et al., "Switch Transformers: Scaling to Trillion Parameter Models"
   - DeepSeek-V2/V3技术报告

3. **RLHF论文**：
   - Ouyang et al., "Training language models to follow instructions with human feedback"
   - Shao et al., "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models" (GRPO)

4. **官方文档**：
   - https://docs.nvidia.com/megatron-core/

---

*本文档基于Megatron-LM v0.15.0编写，持续更新中。*
