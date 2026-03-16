# GitHub 提取：DeepLearing-Interview-Awesome-2024_Agent专题

> 来源仓库：[DeepLearing-Interview-Awesome-2024](https://github.com/315386775/DeepLearing-Interview-Awesome-2024/blob/main/LLMs/Agent.md)
> 本页共整理 3 道题，尽量保留原仓库答案；若原仓库缺少解析，则自动补齐。

## 1. Function call 怎么训练的，怎么微调的？

更详细请查阅[Function call 微调](https://zhuanlan.zhihu.com/p/1971513176509097286 )

## 2. Fucntion call 怎么组织文本的格式喂给模型？

通常采用 System Prompt 注入工具描述，结合多轮对话历史的结构化模板（如 JSON Schema 或 XML）喂给模型。关键点如下：1. 工具描述层：在 System 中，将可用函数的名称、功能、参数类型及必填项按标准格式化。2. 用户意图层：User 角色输入自然语言请求。3. 执行反馈层：若触发调用，模型输出特定格式（如 JSON），系统执行后将结果作为 Tool/Function 角色追加到对话历史中，再次请求模型生成最终回复。

## 3. 你做过 Function Call 微调吗？难点是什么？

做过。核心难点在于高质量数据构造、模型幻觉控制以及复杂多步调用的对齐。关键点如下：1. 数据构造：覆盖单步、多步、并行调用及异常处理（如参数缺失）的数据稀缺，常依赖强模型蒸馏与人工校验。2. 幻觉控制：模型易捏造函数名或参数类型错误，需在 Loss 计算时对工具调用 token 增加权重，或结合约束解码。3. 状态保持：多步调用中模型需准确判断继续调用还是返回答案，易陷入死循环或上下文遗忘。
