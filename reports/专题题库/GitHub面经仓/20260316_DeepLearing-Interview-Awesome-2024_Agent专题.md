# GitHub 提取：DeepLearing-Interview-Awesome-2024_Agent专题

> 来源仓库：[DeepLearing-Interview-Awesome-2024](https://github.com/315386775/DeepLearing-Interview-Awesome-2024/blob/main/LLMs/Agent.md)
> 本页共整理 3 道题，尽量保留原仓库答案；若原仓库缺少解析，则自动补齐。

## 1. Function call 怎么训练的，怎么微调的？

更详细请查阅[Function call 微调](https://zhuanlan.zhihu.com/p/1971513176509097286 )

## 2. Fucntion call 怎么组织文本的格式喂给模型？

结论：Function Call 通常通过特定的 Prompt 模板，将函数签名（如 JSON Schema）与用户查询拼接后作为上下文喂给模型。
关键点：
1. **系统指令 (System Prompt)**：明确告知模型具备工具调用能力，并规定触发调用时的输出格式（通常为标准 JSON）。
2. **工具描述 (Tools Definition)**：将函数名、功能描述、参数列表（类型、描述、是否必填）转化为 JSON Schema 或 XML/Markdown 格式注入。
3. **用户输入 (User Query)**：追加用户的自然语言请求。
4. **特殊 Token 隔离**：许多开源模型（如 Qwen、ChatGLM）在微调时引入了 `<|tool_call|>` 等特殊 Token，组织文本时需严格对齐该模型的官方模板，以激活其内在的工具调用能力。

## 3. 你做过 Function Call 微调吗？难点是什么？

结论：做过。Function Call 微调的核心难点在于高质量训练数据的构造，以及模型在“调用时机”与“参数提取”上的幻觉控制。
关键点：
1. **数据构造与多样性**：需构建包含单工具、多工具路由、多轮调用及“拒绝调用”（不需要工具的闲聊）的混合数据集。人工标注成本高，用强模型蒸馏易引入格式噪声。
2. **参数幻觉与格式对齐**：模型极易捏造用户未提供的参数，或输出的 JSON 结构不符合 Schema 要求（如类型错误、漏掉必填项），导致解析失败。
3. **调用时机误判**：在多工具场景下容易选错工具，或在普通对话中强行触发工具（False Positive）。
4. **灾难性遗忘**：注入 FC 能力时，若数据配比不当，极易导致模型原有的通用对话和逻辑推理能力下降。


