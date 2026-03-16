# LLMs_interview_notes：四、大模型（LLMs）langchain 面

> 来源分组：LLMs_interview_notes
> 本页题目数：10
> 每题均包含基础知识补充、详细解答和案例模拟。

## 大模型（LLMs）langchain 面

### 1. 一、什么是 LangChain?

- 主标签：Agent与工具调用
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / 大模型（LLMs）langchain 面 / 未知](https://articles.zsxq.com/id_ve2dgaiqrjzv.html)

### 基础知识补充

- LangChain是用于开发由大语言模型驱动的应用程序框架。
- 核心价值在于将LLM与外部数据源和计算工具进行连接。
- 提供标准化组件如提示词、链、代理和记忆模块。

### 详细解答

LangChain 是一个开源的编排框架，旨在简化由大语言模型（LLM）驱动的应用程序的开发。它的核心结论是：LLM 本身只是一个推理引擎，而 LangChain 赋予了它“记忆”、“工具”和“执行逻辑”。 在原理上，它通过抽象底层 LLM API 的差异，提供了一套标准化的组件。开发者可以通过 Prompt Templates 规范输入，利用 Chains 将多个处理步骤串联，借助 Agents 让模型自主决定调用何种外部工具（如搜索引擎、数据库），并通过 Memory 模块维持多轮对话的上下文。在工程权衡上，LangChain 极大地提升了开发效率，降低了构建复杂 AI 应用的门槛，但其高度封装也可能导致在极高并发或定制化极强的场景下，调试困难且存在一定的性能开销。

### 案例模拟

业务案例模拟：在构建企业级智能客服时，单纯的 LLM 无法回答公司内部的最新政策。我们引入 LangChain 框架，首先使用 Document Loaders 和 Vector Stores 将企业知识库向量化；然后构建一个 RetrievalQA Chain，当用户提问时，先检索相关文档，再将文档内容作为上下文拼接到 Prompt 中交给 LLM 生成回答。这有效解决了大模型的幻觉问题和数据时效性问题。

### 2. 二、LangChain 包含哪些 核心概念？

- 主标签：Agent与工具调用
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / 大模型（LLMs）langchain 面 / 未知](https://articles.zsxq.com/id_ve2dgaiqrjzv.html)

### 基础知识补充

- Model I/O负责管理大语言模型的输入提示与输出解析。
- Data Connection实现外部数据的加载、转换、存储与检索。
- Agent机制允许模型根据当前状态自主选择并调用外部工具。

### 详细解答

LangChain 的核心概念围绕着如何让大模型更好地与外部世界交互而设计，主要包含六大模块。结论上，这些概念构成了从数据输入到逻辑执行的完整生命周期。 具体包括：1. Model I/O（模型输入输出）：涵盖 Prompt 模板、LLM 接口封装及 Output Parsers（输出解析器）；2. Data Connection（数据连接）：包含文档加载器、文本分割器、向量数据库接口及检索器，是 RAG 的基础；3. Chains（链）：将多个组件组合成特定业务逻辑的执行序列；4. Agents（代理）：赋予大模型决策权，根据任务动态调用 Tools（工具）；5. Memory（记忆）：管理对话历史，使无状态的 LLM 具备上下文感知能力；6. Callbacks（回调）：用于日志记录、监控和流式输出。工程上，这些模块解耦设计，支持灵活替换。

### 案例模拟

面试官追问：“在这些核心概念中，Chain 和 Agent 的本质区别是什么？” 回答：“Chain 是一种硬编码的确定性工作流，执行顺序在代码编写时就已固定，例如先检索数据再生成回答，适合流程明确的任务。而 Agent 是非确定性的，它将 LLM 作为推理引擎，根据用户的输入动态决定执行哪些步骤、调用哪些工具。Agent 更加灵活强大，但也更容易出错，对底层 LLM 的推理和指令遵循能力要求更高。”

### 3. 2.1 LangChain 中 Components and Chains 是什么？

- 主标签：Agent与工具调用
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / 大模型（LLMs）langchain 面 / 未知](https://articles.zsxq.com/id_ve2dgaiqrjzv.html)

### 基础知识补充

- Components是执行单一特定任务的模块化基础构建块。
- Chains是将多个组件按特定顺序组合而成的执行流水线。
- LLMChain是最基础的链，组合了提示词模板与语言模型。

### 详细解答

在 LangChain 中，Components（组件）和 Chains（链）是构建复杂应用的基础架构。结论是：组件是提供单一功能的积木，而链是将积木拼装成有业务价值的工作流。 Components 包含了 LLM 包装器、Prompt 模板、输出解析器等，它们被设计为高度模块化和可复用的单元。然而，单独的组件往往无法完成复杂的业务逻辑。Chains 的作用就是将这些组件串联起来。例如，最常用的 LLMChain 接收用户输入，将其格式化为 Prompt，传递给 LLM，然后解析输出。更复杂的链（如 SequentialChain）可以将前一个链的输出作为后一个链的输入。工程权衡上，使用 Chains 可以极大地提高代码的可读性和复用性，但在处理高度动态的逻辑时，过于复杂的嵌套链可能会导致状态管理困难，此时通常需要转向 Agent 架构。

### 案例模拟

项目案例模拟：在开发一个自动撰写技术博客的工具时，我们使用了 SequentialChain。第一步是一个 LLMChain，根据用户提供的主题生成文章大纲；第二步是另一个 LLMChain，它接收第一步生成的大纲作为输入，逐个章节生成详细内容；第三步使用一个语法检查的 Chain 对全文进行润色。通过将大任务拆解为多个串联的 Chain，显著提升了最终生成文章的质量和逻辑连贯性。

### 4. 2.2 LangChain 中 Prompt Templates and Values 是什么？

- 主标签：Agent与工具调用
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / 大模型（LLMs）langchain 面 / 未知](https://articles.zsxq.com/id_ve2dgaiqrjzv.html)

### 基础知识补充

- Prompt Templates是生成动态提示词的可复用参数化模板。
- Prompt Values是模板填充后传递给模型的最终实例化对象。
- 支持Few-Shot模板以注入示例提升模型特定任务表现。

### 详细解答

Prompt Templates（提示词模板）和 Prompt Values（提示词值）是 LangChain 中管理模型输入的标准化机制。结论是：模板负责定义输入的结构与变量，而值是变量被具体数据填充后的最终产物。 Prompt Templates 允许开发者预先定义一段包含占位符的文本（如“请将以下{language}翻译成中文：{text}”）。在运行时，传入具体的变量字典，模板就会将其渲染为 Prompt Values。Prompt Values 是一个抽象接口，它可以被转换为纯文本（供普通 LLM 使用）或消息列表（供 Chat Model 使用，如包含 SystemMessage、HumanMessage）。工程权衡上，将 Prompt 模板化不仅实现了提示词与业务代码的解耦，方便统一管理和版本控制，还能轻松集成 Few-Shot（少样本）示例，显著提升模型在特定格式输出或垂直领域任务中的稳定性和准确率。

### 案例模拟

面试官追问：“在处理 Chat Model 时，Prompt Template 有什么特殊之处？” 回答：“对于传统的 LLM，Prompt Template 输出的是单一的字符串。但对于 Chat Model（如 GPT-4），输入是结构化的消息序列。因此 LangChain 提供了 ChatPromptTemplate，它允许我们分别定义 System Message（设定角色和规则）、Human Message（用户输入）和 AI Message（历史回复）的模板。这种结构化的模板能更好地利用 Chat Model 的指令遵循能力。”

### 5. 2.3 LangChain 中 Example Selectors 是什么？

- 主标签：Agent与工具调用
- 来源条数：1
- 答案生成方式：近似题匹配：DeepLearing-Interview-Awesome-2024_手撕代码专题:CodeAnything/Reference.md
- 来源：[LLMs_interview_notes / 大模型（LLMs）langchain 面 / 未知](https://articles.zsxq.com/id_ve2dgaiqrjzv.html)

### 基础知识补充

- Example Selectors用于在少样本提示中动态选择最相关的示例。
- 可基于文本相似度、长度或最大边际相关性进行示例筛选。
- 核心目的是在上下文窗口限制内最大化提示词的指导效果。

### 详细解答

Example Selectors（示例选择器）是 LangChain 中用于优化 Few-Shot Prompting（少样本提示）的高级组件。结论是：它通过动态筛选最合适的示例注入到提示词中，以提升模型表现并节省 Token 成本。 在构建复杂应用时，我们可能拥有一个包含数百个示例的庞大样本库。由于 LLM 的上下文窗口有限且 Token 计费，将所有示例都放入 Prompt 是不现实的。Example Selectors 能够根据用户的当前输入，动态挑选出最相关的几个示例。常见的选择策略包括：基于向量数据库的语义相似度选择（SemanticSimilarityExampleSelector）、基于输入长度的动态截断选择（LengthBasedExampleSelector）以及兼顾相关性与多样性的最大边际相关性选择（MMR）。工程权衡上，动态选择示例能显著提高模型在特定任务上的泛化能力，但引入向量检索也会增加系统的延迟和架构复杂度。

### 案例模拟

业务案例模拟：在开发 Text-to-SQL（自然语言转SQL）应用时，我们积累了上千条“自然语言-SQL”的映射对。当用户提问时，我们使用 SemanticSimilarityExampleSelector。它会先将用户的自然语言问题向量化，去向量库中检索出最相似的 3 个历史问答对，然后将这 3 个示例作为 Few-Shot 注入到 Prompt 中。这使得大模型能参考类似表结构的查询逻辑，大幅降低了 SQL 生成的语法错误和幻觉。

### 6. 2.4 LangChain 中 Output Parsers 是什么？

- 主标签：Agent与工具调用
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / 大模型（LLMs）langchain 面 / 未知](https://articles.zsxq.com/id_ve2dgaiqrjzv.html)

### 基础知识补充

- Output Parsers将大模型输出的纯文本转换为结构化数据对象。
- 提供格式化指令注入Prompt以引导模型按特定规则输出。
- 支持JSON、列表、日期等多种数据类型的解析与错误重试机制。

### 详细解答

Output Parsers（输出解析器）是 LangChain 中连接 LLM 输出与下游程序逻辑的关键桥梁。结论是：它解决了大模型输出不可控的问题，使得非结构化的自然语言能够被代码稳定处理。 LLM 本质上只输出字符串，但在实际工程中，我们往往需要 JSON、列表、枚举值或特定的 Pydantic 对象。Output Parsers 的工作原理包含两步：首先，它提供 get_format_instructions() 方法，生成一段明确的格式要求（如 JSON Schema）并自动追加到用户的 Prompt 中，指导模型如何排版输出；其次，它提供 parse() 方法，利用正则表达式或 JSON 解析库，将模型返回的字符串提取并转化为目标数据结构。工程权衡上，虽然解析器能大幅提升系统的自动化程度，但强依赖于模型的指令遵循能力。对于较弱的模型，常需配合 RetryOutputParser（重试解析器）在解析失败时让模型自我修正，这会增加额外的调用延迟。

### 案例模拟

面试官追问：“如果大模型输出的 JSON 格式有轻微错误（比如漏了引号），Output Parser 会怎么处理？” 回答：“LangChain 提供了容错机制。首先可以使用 OutputFixingParser，它在底层解析失败时，会将错误的输出和格式指令再次发送给 LLM，要求 LLM 修复格式错误。如果需要更严格的逻辑校验，可以使用 RetryOutputParser，它不仅传入错误输出，还会传入原始 Prompt，让模型重新思考并生成完全符合格式和逻辑的正确结果。”

### 7. 2.5 LangChain 中 Indexes and Retrievers 是什么？

- 主标签：Agent与工具调用
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / 大模型（LLMs）langchain 面 / 未知](https://articles.zsxq.com/id_ve2dgaiqrjzv.html)

### 基础知识补充

- Indexes是组织和存储文档以便于高效检索的数据结构。
- Retrievers是接收非结构化查询并返回相关文档的通用接口。
- 它们是构建RAG检索增强生成系统的核心数据连接组件。

### 详细解答

Indexes（索引）和 Retrievers（检索器）是 LangChain 中实现数据连接（Data Connection）和 RAG（检索增强生成）的核心组件。结论是：索引负责外部知识的结构化存储，检索器负责在问答时精准召回相关知识。 Indexes 通常涉及文档加载（Document Loaders）、文本分块（Text Splitters）、嵌入向量化（Embeddings）以及存储到向量数据库（Vector Stores）的完整流水线。而 Retrievers 是一个更抽象的接口，它不关心数据是如何存储的，只负责接收用户的 Query 并返回一个 Document 列表。最常见的是基于向量数据库的 VectorStoreRetriever。工程权衡上，简单的向量检索容易遇到语义匹配不准的问题，因此 LangChain 提供了高级检索器，如 MultiQueryRetriever（多查询扩展）、ContextualCompressionRetriever（上下文压缩过滤）等，通过增加一定的计算开销来显著提升召回率和信噪比。

### 案例模拟

项目案例模拟：在构建企业规章制度问答机器人时，单纯的向量检索经常召回大量冗余的段落，导致 LLM 抓不到重点。我们引入了 ContextualCompressionRetriever。当用户提问时，基础检索器先召回 10 个相关文档块，然后压缩器（通常是一个轻量级 LLM 或重排模型）会根据用户的具体问题，对这 10 个文档块进行信息提取和过滤，最终只保留最核心的 3 句话传递给生成模型，大幅降低了 Token 消耗并减少了幻觉。

### 8. 2.6 LangChain 中 Chat Message History 是什么？

- 主标签：Agent与工具调用
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / 大模型（LLMs）langchain 面 / 未知](https://articles.zsxq.com/id_ve2dgaiqrjzv.html)

### 基础知识补充

- 核心组件：用于记录和管理多轮对话历史的存储模块。
- 存储介质：支持内存、Redis、MongoDB等多种持久化后端。
- 消息类型：包含人类消息、AI消息和系统消息等多种标准类型。

### 详细解答

结论：Chat Message History 是 LangChain 中负责记录、存储和检索多轮对话上下文的核心组件。 原理：大模型本身是无状态的，无法记住之前的对话。该组件通过在每次交互时将用户输入和模型输出追加到历史记录中，并在下一次请求时将这些记录作为上下文传递给模型，从而实现多轮对话能力。 工程权衡：在实际工程中，随着对话轮数增加，上下文长度会急剧膨胀，导致Token消耗增加及超出模型上下文窗口限制。因此，通常需要结合 ConversationBufferWindowMemory（滑动窗口）或 ConversationSummaryMemory（历史摘要）等机制，对历史消息进行截断或压缩，以在保留关键信息和控制成本之间取得平衡。

### 案例模拟

面试官追问：如果对话历史非常长，导致Token超限怎么处理？ 回答：在项目中，我通常采用混合策略。对于最近的3-5轮对话，使用滑动窗口直接保留原文以保证短期记忆的准确性；对于更早的对话，则调用轻量级模型生成摘要并存储。此外，还可以将历史对话向量化存入向量数据库，当用户提问时，通过语义检索召回相关的历史片段，从而有效突破上下文窗口限制。

### 9. 2.7 LangChain 中 Agents and Toolkits 是什么？

- 主标签：Agent与工具调用
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / 大模型（LLMs）langchain 面 / 未知](https://articles.zsxq.com/id_ve2dgaiqrjzv.html)

### 基础知识补充

- Agent：基于大模型推理能力动态决定执行步骤的决策引擎。
- Toolkit：为特定任务封装的一组工具集合，如数据库操作工具包。
- 核心机制：通过ReAct等提示框架实现思考、行动与观察的循环。

### 详细解答

结论：Agents 是利用大模型作为推理引擎来动态决定行动路径的代理，而 Toolkits 则是为特定场景预先封装好的一组工具集合。 原理：传统的链式调用（Chain）是硬编码的固定流程，而 Agent 能够根据用户输入，自主判断需要调用哪些工具、以何种顺序调用。Toolkit 比如 SQLDatabaseToolkit，包含了查询表结构、执行SQL、检查错误等多个相关工具，方便 Agent 直接加载使用。 工程权衡：Agent 赋予了系统极大的灵活性，但也引入了不可控性，容易出现幻觉或死循环。在工程实践中，通常需要为 Agent 设置最大迭代次数（max_iterations），并对工具的输入输出进行严格的类型校验和异常捕获，以保证系统的稳定性。

### 案例模拟

面试官追问：在实际业务中，如何防止 Agent 调用工具时发生死循环？ 回答：在开发数据分析 Agent 时，我遇到过模型反复生成错误 SQL 导致死循环的问题。解决方案包括：1. 设置 Agent 的最大执行步数限制；2. 在工具层面增加错误反馈机制，将具体的错误信息（如语法错误）返回给模型，引导其修正；3. 优化 Prompt，明确规定如果连续失败三次则直接返回人工提示，避免无效的 Token 消耗。

## 多轮对话中让AI保持长期记忆的8种优化方式篇

### 10. 二、Agent 如何获取上下文对话信息？

- 主标签：Agent与工具调用
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / 多轮对话中让AI保持长期记忆的8种优化方式篇 / 未知](https://articles.zsxq.com/id_3qgicwcwzjpi.html)

### 基础知识补充

- 记忆模块：通过Memory组件将历史对话注入到当前Prompt中。
- 暂存区：使用Agent Scratchpad记录当前任务的中间推理步骤。
- 检索增强：结合向量数据库召回与当前任务相关的长期历史信息。

### 详细解答

结论：Agent 获取上下文对话信息主要依赖于记忆组件（Memory）和暂存区（Scratchpad）的协同工作。 原理：首先，全局的对话历史通过 Memory 模块（如缓冲记忆或摘要记忆）被提取并格式化，作为系统提示或前置消息传递给大模型，这构成了多轮对话的“长期记忆”。其次，Agent 在执行复杂任务时，会产生一系列的“思考-行动-观察”中间步骤，这些步骤被记录在 Agent Scratchpad 中，作为“短期工作记忆”附加在当前请求中，帮助模型了解当前任务的进展。 工程权衡：将所有上下文直接拼接会导致 Token 激增。实际应用中，需严格区分对话历史和任务执行历史，对对话历史进行摘要或截断，对 Scratchpad 则在任务完成后及时清空，以优化性能和成本。

### 案例模拟

面试官追问：Agent Scratchpad 和 Memory 有什么本质区别？ 回答：Memory 主要是跨越多个用户请求的对话历史，解决的是“用户之前说了什么”的问题；而 Scratchpad 是针对当前单次请求中，Agent 调用工具的中间过程记录，解决的是“我刚才做了哪几步、结果如何”的问题。在项目中，一旦当前任务执行完毕并返回最终结果给用户，Scratchpad 就会被清空，而 Memory 会将这轮完整的问答追加保存。
