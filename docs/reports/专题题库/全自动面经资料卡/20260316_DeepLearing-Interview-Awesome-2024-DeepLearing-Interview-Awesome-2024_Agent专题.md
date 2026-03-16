# DeepLearing-Interview-Awesome-2024：DeepLearing-Interview-Awesome-2024_Agent专题

> 来源分组：DeepLearing-Interview-Awesome-2024
> 本页题目数：3
> 每题均包含基础知识补充、详细解答和案例模拟。

## 原仓库题解

### 1. Function call 怎么训练的，怎么微调的？

- 主标签：Agent与工具调用
- 来源条数：1
- 答案生成方式：原仓库答案
- 来源：[DeepLearing-Interview-Awesome-2024 / 原仓库题解 / 未知](https://github.com/315386775/DeepLearing-Interview-Awesome-2024/blob/main/LLMs/Agent.md)

### 基础知识补充

- 先讲规划与调用
- 再讲状态维护
- 补异常处理策略

### 详细解答

更详细请查阅Function call 微调 答题时还需要补充：这个问题涉及的核心模块、为什么这样设计、以及实际工程里可能遇到的性能、显存或数据质量约束。

### 案例模拟

面试表达可以这样组织：先用一句话回答“Function call 怎么训练的，怎么微调的？”的核心结论，再按“原理 -> 适用场景 -> 工程代价”的顺序展开。若面试官继续追问，就结合你做过的训练、推理、评测或排障经历，说一个具体案例来支撑判断。

### 2. Fucntion call 怎么组织文本的格式喂给模型？

- 主标签：Agent与工具调用
- 来源条数：1
- 答案生成方式：近似题匹配：DeepLearing-Interview-Awesome-2024_Agent专题:LLMs/Agent.md
- 来源：[DeepLearing-Interview-Awesome-2024 / 原仓库题解 / 未知](https://github.com/315386775/DeepLearing-Interview-Awesome-2024/blob/main/LLMs/Agent.md)

### 基础知识补充

- 将工具描述转化为JSON Schema或特定XML格式。
- 在System Prompt中注入可用函数列表及参数说明。
- 采用特殊标识符区分用户输入、模型思考与函数调用。

### 详细解答

结论：Function Call通常通过在System Prompt中注入结构化的工具描述（如JSON Schema），并约定特定的特殊Token或XML标签来组织文本格式。 原理：为了让大模型理解可用工具，需要将函数的名称、功能描述、参数列表及其类型严格定义。主流做法（如OpenAI格式）是将这些信息序列化为JSON Schema，拼接到系统提示词中。当模型决定调用工具时，会生成特定格式的文本（如{"name": "get_weather", "args": {"city": "Beijing"}}）。 工程权衡：JSON格式通用性强，但占用Token较多；XML或YAML格式在某些模型中能节省Token且更容易被正则解析。实际工程中，需要根据基座模型的微调习惯选择对齐的格式，以保证工具调用的准确率和格式合法性。

### 案例模拟

面试官追问：如果模型生成的JSON参数格式错误怎么办？ 回答：在工程实现中，我们会引入重试机制与格式约束。首先，可以使用带有语法约束的解码器（如基于Outlines或Guidance的JSON Schema约束生成），在推理底层强制模型输出合法JSON。其次，如果发生解析错误，会将错误信息作为User Prompt重新喂给模型，让其进行自我纠正，通常重试1-2次即可解决大部分格式问题。

### 3. 你做过 Function Call 微调吗？难点是什么？

- 主标签：Agent与工具调用
- 来源条数：1
- 答案生成方式：近似题匹配：DeepLearing-Interview-Awesome-2024_Agent专题:LLMs/Agent.md
- 来源：[DeepLearing-Interview-Awesome-2024 / 原仓库题解 / 未知](https://github.com/315386775/DeepLearing-Interview-Awesome-2024/blob/main/LLMs/Agent.md)

### 基础知识补充

- 数据构造需覆盖多轮调用、参数缺失与异常处理场景。
- 灾难性遗忘可能导致模型通用对话能力大幅下降。
- 幻觉问题表现为捏造不存在的函数或编造参数值。

### 详细解答

结论：Function Call微调的核心难点在于高质量多样化数据的构造、通用能力的保持以及对工具幻觉的抑制。 原理：在微调时，模型不仅要学会何时调用工具，还要学会何时不调用（拒答或正常对话）。难点一：数据分布极难平衡。如果工具调用数据过多，模型会产生“工具依赖”，对普通问候也尝试调用工具；难点二：参数提取的准确性。对于复杂嵌套的JSON参数，模型容易遗漏必填项或捏造上下文中未提供的信息；难点三：多轮状态跟踪。在多次工具调用交替中，模型容易丢失早期的上下文信息。 工程权衡：通常采用混合训练策略，将Function Call数据与通用SFT数据按特定比例（如1:5）混合。同时，引入负样本（如提供工具但用户问题不需要工具）来增强模型的判断力，牺牲少量训练时间换取鲁棒性。

### 案例模拟

面试官追问：如何解决模型在不需要调用工具时强行调用的问题？ 回答：我们在微调数据中专门构造了约20%的“负样本”。这些样本在System Prompt中提供了丰富的工具列表，但用户的提问完全是闲聊或可以通过常识直接回答。通过在这些样本的Target中要求模型直接输出自然语言回复而不触发标签，有效降低了模型的工具调用误触发率。同时，在推理时适当调低温度参数也有助于减少这种幻觉。
