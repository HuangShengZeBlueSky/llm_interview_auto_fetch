# LLMs_interview_notes 提取：四、大模型（LLMs）langchain 面

> 来源仓库：https://github.com/km1994/LLMs_interview_notes
> 说明：该仓库大量内容以“题目目录 + 外链答案”形式存在，本页保留全量题目清单，并优先复用本仓库现成答案，再补充专题级总结。

## 专题总结

## 本专题考什么
- **大模型应用工程化能力**：考察候选人是否具备将 LLM 从“裸模型”落地为实际应用（如 RAG、Agent）的开发经验。
- **框架底层逻辑与架构设计**：评估对 LangChain 模块化、解耦设计的理解，而非仅仅停留在“调包”层面。
- **状态管理与复杂任务编排**：重点考察在多轮对话中如何管理上下文（Memory），以及如何利用大模型进行推理与工具调用（Agent 机制）。

## 答题主线
- **宏观定位 -> 微观组件**：先定性 LangChain 是“胶水层”和“编排框架”，再拆解其核心模块（Model I/O、Retrieval、Chain、Agent、Memory）。
- **痛点驱动回答**：在解释概念时，结合 LLM 的原生缺陷来答。例如：“因为 LLM 是无状态的，所以需要 Memory 模块”；“因为 LLM 存在幻觉且无法与外部交互，所以需要 Agent 和 Tools”。
- **深入底层机制**：对于具体问题（如上下文获取），要能说透 Prompt 组装的过程（如 `MessagesPlaceholder` 和 `Agent Scratchpad` 的拼接原理），展现资深开发者的代码级理解。

## 代表题答法

### 1. 什么是 LangChain?
**答题要点：**
LangChain 是一个用于开发由大语言模型（LLMs）驱动的应用程序的开源编排框架。它的核心价值在于**标准化**和**组件化**。
- **定位**：它是 LLM 与外部数据、外部工具之间的“胶水层”。
- **解决的痛点**：原生 LLM 只能做文本补全，缺乏记忆、无法访问私有数据、无法执行动作。LangChain 提供了一套标准接口，极大地降低了开发 RAG（检索增强生成）和 Agent（智能体）的门槛。
- **工程价值**：通过统一的 API 封装，开发者可以无缝切换底层大模型（如从 OpenAI 切换到开源的 Llama），而无需重写业务逻辑代码。

### 2. LangChain 包含哪些核心概念？
**答题要点：**
LangChain 的架构由几个高度解耦的核心组件构成，建议按数据流向或功能分类回答：
1. **Model I/O（模型输入输出）**：包含 Prompts（提示词模板）、Language Models（大模型接口，分 LLM 和 ChatModel）、Output Parsers（输出解析器，将文本转为结构化数据）。
2. **Retrieval（数据连接/检索）**：用于 RAG 场景，包含 Document Loaders（文档加载）、Text Splitters（文本切分）、Embedding Models（向量化）和 Vector Stores（向量数据库）。
3. **Chains（链）**：将多个组件（如 Prompt + LLM + Parser）串联起来的执行逻辑，是 LangChain 最基础的编排单元。
4. **Memory（记忆）**：负责在多轮对话中存储和读取历史状态，解决 LLM 无状态的问题。
5. **Agents（智能体）**：将 LLM 作为推理引擎（Reasoning Engine），根据用户输入动态决定调用哪些 Tools（工具），并观察结果直到完成任务（如 ReAct 模式）。
6. **Callbacks（回调系统）**：用于日志记录、监控、流式输出（Streaming）和调试的钩子机制。

### 3. Agent 如何获取上下文对话信息？
**答题要点：**
Agent 获取上下文本质上是**将历史状态动态注入到当前 Prompt 中**的过程。需要分“对话历史”和“推理轨迹”两部分来答：
1. **对话历史（Conversation History）的获取**：
   - **存储介质**：通过 `Memory` 组件（如 `ConversationBufferMemory` 存全量，或 `ConversationSummaryMemory` 存摘要）在内存或数据库中维护历史消息。
   - **注入机制**：在 Agent 的 Prompt 模板中，通常会预留一个 `MessagesPlaceholder(variable_name="chat_history")`。每次 Agent 执行前，框架会从 Memory 中读取历史记录，将其格式化为 `BaseMessage` 列表，并替换掉占位符。
2. **推理轨迹（Agent Scratchpad）的获取**：
   - **内部上下文**：Agent 在解决复杂问题时，可能需要多次调用工具。中间步骤的“思考（Thought）”、“动作（Action）”和“观察（Observation）”也属于上下文。
   - **注入机制**：LangChain 会在 Prompt 末尾维护一个 `agent_scratchpad` 变量。每次工具执行完毕后，结果会被追加到这个暂存区中，连同用户的原始问题和对话历史一起，再次送入 LLM，直到 LLM 决定输出最终答案（Finish）。

## 大模型（LLMs）langchain 面

- 来源链接：https://articles.zsxq.com/id_ve2dgaiqrjzv.html
- 题目数：9

### 原始题目

1. 一、什么是 LangChain?
2. 二、LangChain 包含哪些 核心概念？
3. 2.1 LangChain 中 Components and Chains 是什么？
4. 2.2 LangChain 中 Prompt Templates and Values 是什么？
5. 2.3 LangChain 中 Example Selectors 是什么？
6. 2.4 LangChain 中 Output Parsers 是什么？
7. 2.5 LangChain 中 Indexes and Retrievers 是什么？
8. 2.6 LangChain 中 Chat Message History 是什么？
9. 2.7 LangChain 中 Agents and Toolkits 是什么？

### 可直接复用的答案

#### 示例 1. 2.3 LangChain 中 Example Selectors 是什么？

compute_error_for_line_given_points(1, 2, [[3,6],[6,9],[12,18]])
```

> 匹配来源：DeepLearing-Interview-Awesome-2024_手撕代码专题:CodeAnything/Reference.md | 匹配分数：0.78

## 多轮对话中让AI保持长期记忆的8种优化方式篇

- 来源链接：https://articles.zsxq.com/id_3qgicwcwzjpi.html
- 题目数：1

### 原始题目

1. 二、Agent 如何获取上下文对话信息？
