# LLMs_interview_notes 提取：十、大模型（LLMs）强化学习面

> 来源仓库：https://github.com/km1994/LLMs_interview_notes
> 说明：该仓库大量内容以“题目目录 + 外链答案”形式存在，本页保留全量题目清单，并优先复用本仓库现成答案，再补充专题级总结。

## 专题总结

## 本专题考什么
- **LLM 训练全链路认知**：考察候选人是否清晰大模型从预训练（Pre-training）、指令微调（SFT）到对齐（Alignment/RLHF）的完整 Pipeline。
- **RLHF 与 PPO 核心机制**：重点考察强化学习在 LLM 中的具体落地，包括 PPO 算法中的四大模型（Actor、Reference、Reward、Critic）的交互与更新机制。
- **奖励模型（RM）的设计**：考察对人类偏好数据的处理、RM 的网络结构、排序损失函数（Ranking Loss）的理解，以及 RM 在整个 RL 闭环中的作用。
- **前沿对齐算法（DPO）**：高频考点，要求深刻理解 DPO（直接偏好优化）的数学直觉、损失函数推导，以及它与传统 RLHF（PPO）在工程实现和训练稳定性上的优劣对比。

## 答题主线
- **宏观定位**：先讲清楚 RLHF 在 LLM 训练中的位置（解决 SFT 阶段存在的“幻觉”和“不符合人类价值观”问题，实现 Alignment）。
- **经典范式（RLHF/PPO）**：按照“收集偏好数据 -> 训练 RM -> PPO 强化学习微调”的三步走逻辑展开。强调 PPO 阶段的 KL 散度惩罚和优势函数（Advantage）。
- **痛点与演进（DPO）**：指出 PPO 训练极其不稳定、显存占用大（需同时加载 4 个模型）的痛点，顺理成章地引出 DPO。
- **DPO 核心逻辑**：讲透 DPO 是如何通过数学等价转换，将“强化学习问题”转化为“分类/对比学习问题”，从而省去显式 RM 和 Critic 模型的。

## 代表题答法

### 问题1：简述 LLM 的经典训练 Pipeline 与 RLHF 的整体流程？
**答题要点：**
1. **经典 Pipeline 分为三阶段**：
   - **Pre-training（预训练）**：海量无标签数据，基于 Next Token Prediction 任务，让模型学习世界知识和语言规律，得到 Base Model。
   - **SFT（监督微调）**：高质量人工构造的问答对，让模型学会遵循指令（Instruction Following），得到 SFT Model。
   - **RLHF（基于人类反馈的强化学习）**：使模型输出符合人类偏好（Helpful, Honest, Harmless）。
2. **RLHF 的三大步骤**：
   - **Step 1**：基于 SFT 模型对同一个 Prompt 生成多个回答，人工进行排序打分。
   - **Step 2**：训练奖励模型（Reward Model, RM），学习人类的偏好打分逻辑。
   - **Step 3**：使用 PPO（近端策略优化）算法，以 RM 的输出作为 Reward，对 SFT 模型进行强化学习微调。

### 问题2：详细介绍大模型 RLHF 中的 PPO 算法步骤？
**答题要点：**
PPO 阶段通常需要同时存在 **4 个模型**：Actor（策略模型，待训练）、Reference（参考模型，冻结）、Reward（奖励模型，冻结）、Critic（价值模型，待训练）。
1. **Rollout（采样阶段）**：Actor 模型接收 Prompt 生成 Response。
2. **Reward（奖励计算）**：将 Prompt + Response 输入 Reward 模型，得到一个标量奖励 $r$。同时，计算 Actor 和 Reference 模型输出概率的 **KL 散度**，作为惩罚项加到奖励中，防止 Actor 偏离初始 SFT 模型太远（防止“奖励黑客”现象）。
3. **Advantage（优势估计）**：Critic 模型评估当前状态的价值（Value），结合实际得到的 Reward，计算优势函数（Advantage），即“当前回答比平均水平好多少”。
4. **Update（模型更新）**：
   - **Actor 更新**：最大化优势函数，同时使用 PPO 的 Clip 机制限制策略更新幅度。
   - **Critic 更新**：通过 MSE Loss 拟合真实的 Return，提高价值预测的准确性。

### 问题3：介绍一下 RM（奖励模型）以及为什么需要它？
**答题要点：**
1. **为什么需要 RM**：强化学习需要环境给出密集的 Reward 信号。人类无法实时、高频地为模型生成的每一个 Token 或句子打分（成本太高且速度太慢），因此需要训练一个 RM 作为“人类偏好的代理（Proxy）”来自动发放奖励。
2. **RM 是什么/怎么训练**：
   - **结构**：通常用 SFT 模型初始化，将最后一层替换为线性层，输出一个标量（Scalar）。
   - **损失函数**：采用 Pairwise Ranking Loss（基于 Bradley-Terry 模型）。输入一个 Prompt 和一对回答 $(y_{chosen}, y_{rejected})$，RM 分别输出得分 $r_c$ 和 $r_r$。Loss 旨在最大化两者差值的 Sigmoid 概率：$\mathcal{L} = -\log \sigma(r_c - r_r)$。

### 问题4：详细对比 DPO 与 RLHF，并介绍 DPO 的损失函数？
**答题要点：**
1. **DPO vs RLHF (PPO)**：
   - **模型数量与显存**：PPO 需要 4 个模型（Actor, Ref, RM, Critic），显存开销巨大；DPO 只需要 2 个模型（Policy, Ref），无需显式的 RM 和 Critic。
   - **训练稳定性**：PPO 涉及复杂的超参调优、Actor-Critic 联合训练，极易崩溃；DPO 将 RL 问题转化为了标准的交叉熵（分类）问题，训练极其稳定。
   - **核心思想**：RLHF 是“先拟合奖励，再最大化奖励”；DPO 是“用策略模型的概率比值隐式地表达奖励”，直接在偏好数据上优化策略。
2. **DPO 的损失函数**：
   - DPO 通过数学推导证明了：最优策略与参考策略的对数概率差，等价于隐式的奖励。
   - 损失函数形式类似于对比学习：
     $\mathcal{L}_{DPO} = -\log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)} \right)$
   - **解释**：$\pi_\theta$ 是当前模型，$\pi_{ref}$ 是参考模型。$y_w$ 是 chosen，$y_l$ 是 rejected。$\beta$ 是控制 KL 惩罚强度的温度超参。该 Loss 的本质是**拉大“模型对 chosen 回答的概率提升”与“对 rejected 回答的概率提升”之间的差距**。

## 大模型（LLMs）强化学习面

- 来源链接：https://articles.zsxq.com/id_20xnfnoprj9s.html
- 题目数：8

### 原始题目

1. 1 简单介绍强化学习？
2. 2 简单介绍一下 RLHF？
3. 3 奖励模型需要和基础模型一致吗？
4. 4 RLHF 在实践过程中存在哪些不足？
5. 5 如何解决 人工产生的偏好数据集成本较高，很难量产问题？
6. 6 如何解决三个阶段的训练（SFT-\>RM-\>PPO）过程较长，更新迭代较慢问题？
7. 7 如何解决 PPO 的训练过程同时存在4个模型（2训练，2推理），对计算资源的要求较高 问题？
8. 8 强化学习跟大语言模型的本质联系是什么？

## 大模型（LLMs）强化学习——RLHF及其变种面

- 来源链接：https://articles.zsxq.com/id_3ct6sw0wouna.html
- 题目数：7

### 原始题目

1. 一、介绍一下 LLM的经典预训练Pipeline？
2. 二、预训练（Pre-training）篇
3. 2.1 具体介绍一下 预训练（Pre-training）？
4. 3.1 具体介绍一下 有监督微调（Supervised Tinetuning）？
5. 3.2 有监督微调（Supervised Tinetuning）的训练数据格式是什么样？
6. 3.3 预训练（Pre-training） vs 有监督微调（Supervised Tinetuning）区别？
7. 4.1 简单介绍一下 对齐（Alignment）？

## 大模型（LLMs）强化学习—— PPO 面

- 来源链接：https://articles.zsxq.com/id_s8kwqw1gowvh.html
- 题目数：6

### 原始题目

1. 一、大语言模型RLHF中的PPO主要分哪些步骤？
2. 二、举例描述一下 大语言模型的RLHF？
3. 3.1 什么是 PPO 中 采样过程？
4. 3.2 介绍一下 PPO 中 采样策略？
5. 3.3 PPO 中 采样策略中，如何评估“收益”？
6. 四、在PPO过程中，reward model的效果上会有什么问题？

## RLHF平替算法DPO篇

- 来源链接：https://articles.zsxq.com/id_mlq44r1p7nob.html
- 题目数：7

### 原始题目

1. 一、DPO vs RLHF？
2. 二、介绍一下 DPO的损失函数？
3. 三、DPO 微调流程 ?
4. 四、说一下 DPO 是如何简化 RLHF 的？
5. 五、DPO的第0步loss是固定的么？如果固定的话，值是多少？
6. 六、DPO是一个on-policy还是off-policy的算法，以及这样的算法有什么优劣？
7. 七、DPO公式是由PPO的objective公式推导过来的，为什么DPO是off-policy算法，而PPO是on-policy算法，到底哪一步推导出了问题？

## reward 篇

- 来源链接：https://articles.zsxq.com/id_vblb0j5qnaxg.html
- 题目数：5

### 原始题目

1. 1 介绍一下 RM模型？
2. 2 为什么需要 RM模型？
3. 3 RM模型训练数据如何构建？
4. 4 reward 模型训练步骤中，为什么这一步骤在标注数据过程中不让人直接打分，而是去标排列序列呢?
5. 5 reward 模型的 loss 是怎么计算的?

## 强化学习在自然语言处理下的应用篇

- 来源链接：https://articles.zsxq.com/id_5tsn84l32eea.html
- 题目数：3

### 原始题目

1. 1.1 介绍一下强化学习？
2. 1.2 介绍一下强化学习 的 状态（States） 和 观测（Observations）？
3. 1.3 强化学习 有哪些 动作空间（Action Spaces），他们之间的区别是什么？
