# Megatron-LM 8卡4090 完整复现计划

> 基于实际硬件资源的全流程学习与复现方案
> 硬件配置：8× NVIDIA RTX 4090 (24GB VRAM each)

---

## 目录

1. [硬件评估与限制分析](#1-硬件评估与限制分析)
2. [环境搭建](#2-环境搭建)
3. [分阶段复现计划](#3-分阶段复现计划)
4. [详细配置与脚本](#4-详细配置与脚本)
5. [预期问题与解决方案](#5-预期问题与解决方案)
6. [学习检查清单](#6-学习检查清单)

---

## 1. 硬件评估与限制分析

### 1.1 4090 硬件规格

| 参数 | 规格 | 对训练的影响 |
|------|------|-------------|
| **显存** | 24GB GDDR6X | 限制模型大小，8B模型需要优化 |
| **FP32算力** | 82.6 TFLOPS | 足够 |
| **BF16算力** | 165.2 TFLOPS | 推荐使用 |
| **FP8算力** | 330.3 TFLOPS | 可用但不如H100 |
| **显存带宽** | 1008 GB/s | 良好 |
| **NVLink** | ❌ 无 | TP效率低，建议TP≤2 |
| **PCIe** | 4.0 x16 (~32GB/s) | 跨卡通信瓶颈 |

### 1.2 关键限制

| 限制 | 影响 | 解决方案 |
|------|------|---------|
| 无NVLink | TP通信开销大 | 使用TP=1或TP=2，优先DP |
| 24GB显存 | 大模型放不下 | 使用分布式优化器+重计算 |
| PCIe带宽 | AllReduce慢 | 使用梯度累积减少通信频率 |

### 1.3 可训练模型规模估算

| 模型 | 参数量 | 显存需求(训练) | 4090可行性 | 并行策略 |
|------|--------|---------------|-----------|---------|
| **Qwen3 4B** | 4B | ~20-28GB | ✅ 推荐 | TP=1, DP=8 |
| **Qwen3 8B** | 8B | ~28-40GB | ⚠️ 需优化 | TP=2, DP=4 |
| Qwen3 32B | 32B | ~100GB+ | ❌ 不可行 | 需要更多GPU |

---

## 2. 环境搭建

### 2.1 方案选择

**推荐方案：NVIDIA PyTorch Container**

```bash
# 拉取容器（包含PyTorch、CUDA、cuDNN、TransformerEngine）
docker pull nvcr.io/nvidia/pytorch:25.06-py3

# 启动容器
docker run --gpus all -it --rm \
    --shm-size=64g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v /data/home/yizhou/Megatron-LM:/workspace/Megatron-LM \
    -v /data/home/yizhou/data:/workspace/data \
    nvcr.io/nvidia/pytorch:25.06-py3
```

### 2.2 依赖安装

```bash
# 进入容器后
cd /workspace/Megatron-LM

# 安装Megatron-LM
pip install -e .

# 安装RL额外依赖
pip install flask-restful uvloop datasets evaluate wandb

# 验证安装
python -c "import megatron; print('Megatron installed successfully')"
```

### 2.3 环境验证

```bash
# 运行最简示例验证环境
torchrun --nproc_per_node=2 examples/run_simple_mcore_train_loop.py

# 预期输出：训练循环正常运行，loss下降
```

---

## 3. 分阶段复现计划

### 阶段0：环境验证与基础概念（Day 1）

**目标**：验证环境，理解Megatron基本概念

**任务清单**：
- [ ] 搭建Docker环境
- [ ] 安装Megatron-LM及依赖
- [ ] 运行最简训练示例
- [ ] 阅读README和核心文档
- [ ] 理解项目目录结构

**验证命令**：
```bash
# 1. 验证GPU
nvidia-smi

# 2. 验证PyTorch CUDA
python -c "import torch; print(torch.cuda.device_count()); print(torch.cuda.get_device_name(0))"

# 3. 运行最简示例
torchrun --nproc_per_node=2 examples/run_simple_mcore_train_loop.py

# 4. 估算显存
python tools/report_theoretical_memory.py \
    --num-layers 12 \
    --hidden-size 512 \
    --num-attention-heads 8 \
    --seq-length 1024 \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1
```

**学习产出**：
- 环境验证通过
- 理解Megatron的核心概念（TP/PP/DP）
- 了解显存计算方法

---

### 阶段1：GPT预训练基础（Day 2-3）

**目标**：掌握Megatron的GPT预训练流程

**任务清单**：
- [ ] 准备小规模预训练数据
- [ ] 运行345M GPT模型预训练
- [ ] 理解数据预处理流程
- [ ] 掌握并行策略配置
- [ ] 学习checkpoint保存/加载

**Step 1: 数据准备**

```bash
# 创建示例数据
cat > /workspace/data/sample.jsonl << 'EOF'
{"text": "The transformer architecture has revolutionized natural language processing. It uses self-attention mechanisms to process sequences in parallel."}
{"text": "Large language models are trained on vast amounts of text data. They learn patterns and can generate coherent text."}
{"text": "Distributed training enables the training of models that are too large for a single GPU. Various parallelism strategies are used."}
EOF

# 预处理数据
python tools/preprocess_data.py \
    --input /workspace/data/sample.jsonl \
    --output-prefix /workspace/data/sample_processed \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model Qwen/Qwen3-4B \
    --workers 2 \
    --append-eod
```

**Step 2: 运行345M GPT预训练**

```bash
# 创建训练脚本
cat > /workspace/scripts/pretrain_gpt_345m.sh << 'EOF'
#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1

torchrun --nproc_per_node=8 --nnodes=1 pretrain_gpt.py \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --num-layers 12 \
    --hidden-size 512 \
    --num-attention-heads 8 \
    --seq-length 1024 \
    --max-position-embeddings 1024 \
    --micro-batch-size 4 \
    --global-batch-size 32 \
    --lr 0.00015 \
    --min-lr 0.00001 \
    --lr-decay-style cosine \
    --lr-warmup-fraction 0.01 \
    --weight-decay 1e-2 \
    --clip-grad 1.0 \
    --bf16 \
    --optimizer adam \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --use-distributed-optimizer \
    --overlap-grad-reduce \
    --overlap-param-gather \
    --recompute-granularity selective \
    --recompute-activations \
    --recompute-modules core_attn \
    --train-iters 1000 \
    --log-interval 10 \
    --save-interval 500 \
    --eval-interval 200 \
    --ckpt-format torch_dist \
    --save /workspace/checkpoints/gpt_345m \
    --data-path /workspace/data/sample_processed \
    --vocab-file /workspace/data/tokenizer/vocab.json \
    --merge-file /workspace/data/tokenizer/merges.txt \
    --split 98,1,1 \
    --dataset-impl mmap
EOF

chmod +x /workspace/scripts/pretrain_gpt_345m.sh
bash /workspace/scripts/pretrain_gpt_345m.sh
```

**预期输出**：
```
iteration     10/  1000 | consumed samples: 320 | elapsed time per iteration (ms): 1234.5 | learning rate: 1.5e-04 | loss: 10.234
iteration     20/  1000 | consumed samples: 640 | elapsed time per iteration (ms): 1100.2 | learning rate: 1.5e-04 | loss: 9.876
...
```

**Step 3: 理解并行策略**

```bash
# 尝试不同并行策略，对比性能

# 配置1: 纯DP (baseline)
--tensor-model-parallel-size 1
--pipeline-model-parallel-size 1

# 配置2: TP=2, DP=4
--tensor-model-parallel-size 2
--pipeline-model-parallel-size 1

# 配置3: PP=2, DP=4
--tensor-model-parallel-size 1
--pipeline-model-parallel-size 2

# 记录每种配置的训练速度和MFU
```

**学习产出**：
- 成功运行GPT预训练
- 理解数据预处理流程
- 掌握并行策略配置
- 理解MFU指标

---

### 阶段2：模型转换与SFT（Day 4-5）

**目标**：学习模型转换和SFT流程

**任务清单**：
- [ ] 下载Qwen3-4B模型
- [ ] 转换为Megatron格式
- [ ] 运行SFT训练
- [ ] 理解checkpoint机制

**Step 1: 下载模型**

```bash
# 安装huggingface-hub
pip install huggingface-hub

# 下载Qwen3-4B
huggingface-cli download Qwen/Qwen3-4B --local-dir /workspace/models/qwen3-4b-hf
```

**Step 2: 转换为Megatron格式**

```bash
# HuggingFace -> Megatron转换
python tools/checkpoint/convert.py \
    --bf16 \
    --model-type GPT \
    --loader llama_mistral \
    --saver core \
    --target-tensor-parallel-size 1 \
    --target-pipeline-parallel-size 1 \
    --checkpoint-type hf \
    --load-dir /workspace/models/qwen3-4b-hf \
    --save-dir /workspace/checkpoints/qwen3-4b-megatron \
    --tokenizer-model /workspace/models/qwen3-4b-hf
```

**Step 3: 准备SFT数据**

```bash
# 创建SFT数据（对话格式）
cat > /workspace/data/sft_sample.jsonl << 'EOF'
{"conversations": [{"from": "human", "value": "What is machine learning?"}, {"from": "gpt", "value": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed."}]}
{"conversations": [{"from": "human", "value": "Explain gradient descent."}, {"from": "gpt", "value": "Gradient descent is an optimization algorithm that iteratively adjusts parameters to minimize a loss function by moving in the direction of steepest descent."}]}
EOF
```

**Step 4: 运行SFT**

```bash
cat > /workspace/scripts/sft_qwen3_4b.sh << 'EOF'
#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1

CHECKPOINT=/workspace/checkpoints/qwen3-4b-megatron

torchrun --nproc_per_node=8 --nnodes=1 pretrain_gpt.py \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --num-layers 36 \
    --hidden-size 2560 \
    --ffn-hidden-size 9728 \
    --num-attention-heads 32 \
    --group-query-attention \
    --num-query-groups 8 \
    --seq-length 4096 \
    --max-position-embeddings 40960 \
    --micro-batch-size 1 \
    --global-batch-size 8 \
    --lr 2e-5 \
    --min-lr 2e-6 \
    --lr-decay-style cosine \
    --lr-warmup-fraction 0.1 \
    --weight-decay 0.01 \
    --clip-grad 1.0 \
    --bf16 \
    --optimizer adam \
    --adam-beta1 0.9 \
    --adam-beta2 0.999 \
    --use-distributed-optimizer \
    --overlap-grad-reduce \
    --recompute-granularity selective \
    --recompute-activations \
    --recompute-modules core_attn \
    --train-iters 500 \
    --log-interval 10 \
    --save-interval 250 \
    --ckpt-format torch_dist \
    --save /workspace/checkpoints/qwen3-4b-sft \
    --data-path /workspace/data/sft_processed \
    --pretrained-checkpoint $CHECKPOINT \
    --finetune
EOF

chmod +x /workspace/scripts/sft_qwen3_4b.sh
bash /workspace/scripts/sft_qwen3_4b.sh
```

**学习产出**：
- 掌握模型转换流程
- 理解SFT训练配置
- 理解checkpoint机制
- 理解预训练vs SFT的区别

---

### 阶段3：RL/GRPO训练（Day 6-10）⭐ 核心阶段

**目标**：掌握Megatron-RL的GRPO训练流程

**任务清单**：
- [ ] 理解GRPO算法原理
- [ ] 运行Countdown环境（最简单）
- [ ] 运行GSM8K环境（标准数学推理）
- [ ] 分析训练结果
- [ ] 调参优化

**Step 1: 理解GRPO核心概念**

**关键概念**：
```
GRPO = Group Relative Policy Optimization

核心流程：
1. 从prompt生成多个response（group_size个）
2. 计算每个response的reward
3. 用组内相对排序计算优势（无需Critic）
4. 用PPO风格的clip损失更新策略
5. 可选：KL散度约束防止策略退化
```

**Step 2: 准备环境配置**

```bash
# 查看可用的RL环境配置
ls examples/rl/environment_configs/

# 使用GSM8K环境（最经典）
cat examples/rl/environment_configs/gsm8k.yaml
```

**Step 3: 运行Countdown环境（最简单入门）**

```bash
cat > /workspace/scripts/rl_countdown_qwen3_4b.sh << 'EOF'
#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1

CHECKPOINT=/workspace/checkpoints/qwen3-4b-megatron
ENV_CONFIG=examples/rl/environment_configs/countdown.yaml

torchrun --nproc-per-node=8 --nnodes=1 train_rl.py \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --use-mcore-models \
    --transformer-impl transformer_engine \
    --bf16 \
    --te-rng-tracker \
    --attention-backend flash \
    --cuda-graph-impl local \
    --inference-dynamic-batching-num-cuda-graphs 1 \
    --inference-dynamic-batching-buffer-size-gb 8 \
    --data-parallel-random-init \
    --use-distributed-optimizer \
    --num-layers 36 \
    --hidden-size 2560 \
    --ffn-hidden-size 9728 \
    --num-attention-heads 32 \
    --kv-channels 128 \
    --group-query-attention \
    --num-query-groups 8 \
    --max-position-embeddings 40960 \
    --swiglu \
    --normalization RMSNorm \
    --norm-epsilon 1e-6 \
    --qk-layernorm \
    --position-embedding-type rope \
    --rotary-base 1000000 \
    --disable-bias-linear \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model /workspace/models/qwen3-4b-hf \
    --vocab-size 151936 \
    --make-vocab-size-divisible-by 128 \
    --pretrained-checkpoint $CHECKPOINT \
    --seq-length 2048 \
    --inference-max-seq-length 2048 \
    --inference-max-requests 8 \
    --micro-batch-size 1 \
    --global-batch-size 64 \
    --grpo-group-size 4 \
    --grpo-prompts-per-step 16 \
    --grpo-iterations 1 \
    --grpo-clamp-eps-lower 0.2 \
    --grpo-clamp-eps-upper 0.2 \
    --grpo-kl-beta 0.0 \
    --langrl-env-config $ENV_CONFIG \
    --lr 1e-6 \
    --clip-grad 1.0 \
    --weight-decay 0.01 \
    --adam-beta1 0.9 \
    --adam-beta2 0.999 \
    --recompute-granularity selective \
    --recompute-activations \
    --recompute-modules core_attn \
    --log-interval 5 \
    --eval-interval 50 \
    --save-interval 50 \
    --ckpt-format torch_dist \
    --save /workspace/checkpoints/rl_countdown \
    --train-iters 200
EOF

chmod +x /workspace/scripts/rl_countdown_qwen3_4b.sh
bash /workspace/scripts/rl_countdown_qwen3_4b.sh
```

**Step 4: 运行GSM8K环境（标准数学推理）**

```bash
# 将ENV_CONFIG改为gsm8k
ENV_CONFIG=examples/rl/environment_configs/gsm8k.yaml

# 其他参数相同，但可以增加训练迭代数
--train-iters 500
--grpo-group-size 8
--grpo-prompts-per-step 32
```

**Step 5: 监控训练**

```bash
# 启动TensorBoard
tensorboard --logdir /workspace/checkpoints/rl_countdown --port 6006

# 关键指标
# - lm loss: 训练损失
# - rl/kl_term: KL散度
# - rl/entropy_term: 策略熵
# - rl/pi_over_pi_old: 重要性采样比率
# - reward: 奖励值
```

**Step 6: 评估模型**

```bash
# 使用AIME评估
python examples/rl/evaluation/eval_aime.py \
    --model-path /workspace/checkpoints/rl_countdown/checkpoint_latest \
    --output-file /workspace/results/aime_eval.json
```

**学习产出**：
- 深入理解GRPO算法
- 掌握RL训练流程
- 理解奖励设计
- 学会分析训练曲线
- 掌握超参数调优

---

### 阶段4：优化与进阶（Day 11-14）

**目标**：深入理解优化技术，尝试更复杂的场景

**任务清单**：
- [ ] 尝试Qwen3-8B模型（TP=2）
- [ ] 探索序列打包优化
- [ ] 尝试不同KL惩罚系数
- [ ] 分析性能瓶颈
- [ ] 理解工程最佳实践

**进阶实验1：Qwen3-8B GRPO**

```bash
# 需要TP=2来适配显存
--tensor-model-parallel-size 2
--num-layers 36
--hidden-size 4096
--ffn-hidden-size 12288
--num-attention-heads 32
--seq-length 4096
--inference-max-seq-length 4096
--inference-max-requests 4
--inference-dynamic-batching-buffer-size-gb 5
```

**进阶实验2：序列打包**

```bash
# 启用序列打包，提高GPU利用率
--rl-use-sequence-packing
--rl-sequence-packing-max-sequences-per-bin 4
```

**进阶实验3：KL惩罚调优**

```bash
# 尝试不同的KL惩罚系数
--grpo-kl-beta 0.0   # 无KL约束
--grpo-kl-beta 0.01  # 轻度约束
--grpo-kl-beta 0.1   # 中度约束
--grpo-kl-beta 0.5   # 强约束
```

**性能分析**：

```bash
# 使用NVTX profiling
--nvtx-ranges
--profile-step-start 100
--profile-step-end 110

# 使用PyTorch Profiler
--use-pytorch-profiler
```

---

## 4. 详细配置与脚本

### 4.1 核心配置参数说明

#### 并行策略参数

| 参数 | 说明 | 4090推荐值 |
|------|------|-----------|
| `--tensor-model-parallel-size` | TP并行度 | 1或2 |
| `--pipeline-model-parallel-size` | PP并行度 | 1 |
| `--use-distributed-optimizer` | 分布式优化器 | 开启 |
| `--overlap-grad-reduce` | 梯度通信重叠 | 开启 |
| `--overlap-param-gather` | 参数通信重叠 | 开启 |

#### 显存优化参数

| 参数 | 说明 | 效果 |
|------|------|------|
| `--recompute-granularity selective` | 选择性重计算 | 减少~50%激活显存 |
| `--recompute-modules core_attn` | 只重计算attention | 平衡速度和显存 |
| `--bf16` | BF16精度 | 显存减半 |
| `--use-distributed-optimizer` | ZeRO-1 | 优化器状态分片 |

#### GRPO参数

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `--grpo-group-size` | 每个prompt生成的response数 | 4-16 |
| `--grpo-prompts-per-step` | 每步的prompt数 | 16-64 |
| `--grpo-kl-beta` | KL惩罚系数 | 0.0-0.1 |
| `--grpo-clamp-eps-lower` | PPO clip下界 | 0.2 |
| `--grpo-clamp-eps-upper` | PPO clip上界 | 0.2 |

#### 推理参数

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `--inference-max-seq-length` | 推理最大长度 | 2048-4096 |
| `--inference-max-requests` | 最大并发请求数 | 4-16 |
| `--inference-dynamic-batching-buffer-size-gb` | 推理缓冲区大小 | 5-15GB |

### 4.2 完整脚本模板

```bash
#!/bin/bash
# Megatron-RL GRPO Training Script for 8x4090
# Usage: bash train_grpo.sh [model_size] [env_name]

set -e

# 参数
MODEL_SIZE=${1:-"4b"}  # 4b or 8b
ENV_NAME=${2:-"gsm8k"}  # countdown, gsm8k, dapo

# 路径配置
MEGATRON_DIR=/workspace/Megatron-LM
MODEL_DIR=/workspace/models/qwen3-${MODEL_SIZE}-hf
CHECKPOINT_DIR=/workspace/checkpoints/qwen3-${MODEL_SIZE}-megatron
OUTPUT_DIR=/workspace/checkpoints/rl_${ENV_NAME}_${MODEL_SIZE}
ENV_CONFIG=${MEGATRON_DIR}/examples/rl/environment_configs/${ENV_NAME}.yaml

# 根据模型大小设置参数
if [ "$MODEL_SIZE" = "4b" ]; then
    NUM_LAYERS=36
    HIDDEN_SIZE=2560
    FFN_HIDDEN_SIZE=9728
    NUM_HEADS=32
    NUM_KV_HEADS=8
    TP_SIZE=1
    SEQ_LEN=4096
    MICRO_BATCH=1
    GLOBAL_BATCH=64
    INFERENCE_BUFFER_GB=8
elif [ "$MODEL_SIZE" = "8b" ]; then
    NUM_LAYERS=36
    HIDDEN_SIZE=4096
    FFN_HIDDEN_SIZE=12288
    NUM_HEADS=32
    NUM_KV_HEADS=8
    TP_SIZE=2
    SEQ_LEN=4096
    MICRO_BATCH=1
    GLOBAL_BATCH=32
    INFERENCE_BUFFER_GB=5
else
    echo "Unknown model size: $MODEL_SIZE"
    exit 1
fi

export CUDA_DEVICE_MAX_CONNECTIONS=1

cd $MEGATRON_DIR

torchrun --nproc-per-node=8 --nnodes=1 train_rl.py \
    --tensor-model-parallel-size $TP_SIZE \
    --pipeline-model-parallel-size 1 \
    --use-mcore-models \
    --transformer-impl transformer_engine \
    --bf16 \
    --te-rng-tracker \
    --attention-backend flash \
    --cuda-graph-impl local \
    --inference-dynamic-batching-num-cuda-graphs 1 \
    --inference-dynamic-batching-buffer-size-gb $INFERENCE_BUFFER_GB \
    --data-parallel-random-init \
    --use-distributed-optimizer \
    --overlap-grad-reduce \
    --overlap-param-gather \
    --num-layers $NUM_LAYERS \
    --hidden-size $HIDDEN_SIZE \
    --ffn-hidden-size $FFN_HIDDEN_SIZE \
    --num-attention-heads $NUM_HEADS \
    --group-query-attention \
    --num-query-groups $NUM_KV_HEADS \
    --max-position-embeddings 40960 \
    --swiglu \
    --normalization RMSNorm \
    --norm-epsilon 1e-6 \
    --qk-layernorm \
    --position-embedding-type rope \
    --rotary-base 1000000 \
    --disable-bias-linear \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model $MODEL_DIR \
    --vocab-size 151936 \
    --make-vocab-size-divisible-by 128 \
    --pretrained-checkpoint $CHECKPOINT_DIR \
    --seq-length $SEQ_LEN \
    --inference-max-seq-length $SEQ_LEN \
    --inference-max-requests 8 \
    --micro-batch-size $MICRO_BATCH \
    --global-batch-size $GLOBAL_BATCH \
    --grpo-group-size 8 \
    --grpo-prompts-per-step 32 \
    --grpo-iterations 1 \
    --grpo-clamp-eps-lower 0.2 \
    --grpo-clamp-eps-upper 0.2 \
    --grpo-kl-beta 0.0 \
    --langrl-env-config $ENV_CONFIG \
    --lr 1e-6 \
    --clip-grad 1.0 \
    --weight-decay 0.01 \
    --adam-beta1 0.9 \
    --adam-beta2 0.999 \
    --recompute-granularity selective \
    --recompute-activations \
    --recompute-modules core_attn \
    --empty-unused-memory-level 2 \
    --log-interval 10 \
    --eval-interval 100 \
    --save-interval 100 \
    --ckpt-format torch_dist \
    --save $OUTPUT_DIR \
    --train-iters 1000 \
    2>&1 | tee ${OUTPUT_DIR}/train.log
```

---

## 5. 预期问题与解决方案

### 5.1 显存不足 (OOM)

**症状**：
```
RuntimeError: CUDA out of memory. Tried to allocate X MiB
```

**解决方案（按优先级）**：

1. **减小推理缓冲区**
   ```bash
   --inference-dynamic-batching-buffer-size-gb 5  # 从20降到5
   ```

2. **减小序列长度**
   ```bash
   --seq-length 2048  # 从4096降到2048
   --inference-max-seq-length 2048
   ```

3. **启用重计算**
   ```bash
   --recompute-granularity selective
   --recompute-activations
   --recompute-modules core_attn
   ```

4. **使用TP=2**
   ```bash
   --tensor-model-parallel-size 2
   ```

5. **减小group size**
   ```bash
   --grpo-group-size 4  # 从8降到4
   --grpo-prompts-per-step 16
   ```

### 5.2 训练速度慢

**症状**：每步耗时过长

**解决方案**：

1. **启用CUDA Graphs**
   ```bash
   --cuda-graph-impl local
   --inference-dynamic-batching-num-cuda-graphs 1
   ```

2. **启用通信重叠**
   ```bash
   --overlap-grad-reduce
   --overlap-param-gather
   ```

3. **增大batch size**
   ```bash
   --micro-batch-size 2  # 如果显存允许
   --global-batch-size 128
   ```

4. **使用Flash Attention**
   ```bash
   --attention-backend flash
   ```

### 5.3 通信错误

**症状**：
```
NCCL error: unhandled system error
```

**解决方案**：

1. **设置NCCL环境变量**
   ```bash
   export NCCL_DEBUG=INFO
   export NCCL_IB_DISABLE=1  # 如果没有InfiniBand
   export NCCL_P2P_DISABLE=1  # 如果PCIe P2P有问题
   ```

2. **减小TP并行度**
   ```bash
   --tensor-model-parallel-size 1  # 避免TP通信
   ```

### 5.4 Checkpoint加载失败

**症状**：
```
KeyError: 'model.language_model.embedding.weight'
```

**解决方案**：

1. **检查checkpoint格式**
   ```bash
   --auto-detect-ckpt-format
   ```

2. **确认TP/PP一致性**
   ```bash
   # 加载时的TP/PP必须和保存时一致
   --target-tensor-parallel-size 1
   --target-pipeline-parallel-size 1
   ```

---

## 6. 学习检查清单

### 阶段0：环境验证 ✅

- [ ] Docker环境搭建成功
- [ ] GPU被正确识别（8×4090）
- [ ] Megatron-LM安装成功
- [ ] 最简训练示例运行成功
- [ ] 理解显存计算方法

### 阶段1：GPT预训练 ✅

- [ ] 数据预处理流程掌握
- [ ] 345M模型训练成功
- [ ] 理解并行策略配置
- [ ] 理解MFU指标含义
- [ ] checkpoint保存/加载成功

### 阶段2：模型转换与SFT ✅

- [ ] 模型下载成功
- [ ] HF -> Megatron转换成功
- [ ] SFT训练运行成功
- [ ] 理解预训练vs SFT区别

### 阶段3：RL/GRPO训练 ⭐

- [ ] 理解GRPO算法原理
- [ ] Countdown环境运行成功
- [ ] GSM8K环境运行成功
- [ ] 能够分析训练曲线
- [ ] 理解奖励设计
- [ ] 掌握超参数调优

### 阶段4：优化与进阶 ✅

- [ ] 尝试Qwen3-8B模型
- [ ] 理解序列打包优化
- [ ] 掌握性能分析方法
- [ ] 理解工程最佳实践

---

## 附录：关键文件索引

| 文件 | 说明 |
|------|------|
| `train_rl.py` | RL训练主入口 |
| `pretrain_gpt.py` | GPT预训练入口 |
| `examples/rl/model_configs/` | 模型配置 |
| `examples/rl/environment_configs/` | 环境配置 |
| `megatron/rl/rl_utils.py` | GRPO核心实现 |
| `megatron/training/training.py` | 训练循环 |
| `megatron/core/tensor_parallel/` | TP实现 |
| `megatron/core/pipeline_parallel/` | PP实现 |

---

*本计划基于Megatron-LM v0.15.0和8×RTX 4090制定*
*预计完成时间：2周*
