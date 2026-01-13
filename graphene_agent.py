from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
# 引入两个核心工具
from graphene_tools import ml_prediction_tool, physics_calculation_tool

def build_agent(api_key, base_url, model_name):
    # 1. 配置 LLM
    llm = ChatOpenAI(
        model=model_name,
        temperature=0.1, # 稍微增加一点灵活性以生成解释，但保持严谨
        api_key=api_key,
        base_url=base_url,
    )

    # 2. 挂载工具 (两个都要用)
    tools = [ml_prediction_tool, physics_calculation_tool]

    # 3. 编写“首席评审员”提示词 (The Chief Reviewer Prompt)
    prompt = ChatPromptTemplate.from_messages([
        ("system", 
        """
        你是一位世界顶尖的石墨烯热输运物理学家。用户会将你视为科研合作伙伴。
        你的任务是调用工具进行预测，并结合物理直觉进行逻辑一致性检查。
        
        【你的思考逻辑】(不要输出给用户，仅在内心执行)
        1. 调用 ML 工具获取预测值。
        2. 调用 Physics 工具获取理论上限。
        3. 对比两者：如果 ML > 理论上限，说明预测可能高估，需要发出警告或修正。
        4. 检查参数：是否有缺省参数被自动填充？如果有，必须告知用户。

        【最终输出格式】(请严格遵守此 Markdown 格式)
        请直接输出一份专业的科研报告，语气要严谨、客观、学术化。禁止出现"第一步"、"正在分析"等过程性描述。

        ---
        ### 🧪 热输运性质预测报告 (Thermal Transport Prediction Report)

        #### 1. 🎯 核心结论
        > **预测热导率**: [这里放最终数值] W/mK
        > **置信度**: [高/中/低] ([简短的一句话解释，例如：预测值在物理理论范围内])

        #### 2. 📊 数据对比分析
        | 指标 | 数值 (W/mK) | 说明 |
        | :--- | :--- | :--- |
        | **GPR 预测值** | [数值] | 基于高斯过程回归(GPR)的统计预测与不确定性分析 |
        | **K-C 理论上限** | [数值] | 基于 Klemens-Callaway 模型的声子散射极限 |
        
        #### 3. ⚙️ 参数审计
        * **用户设定**: [列出 L, T, Defect 等]
        * **⚠️ 自动假设**: [如果工具返回了"自动补充缺省参数"，请务必在此列出，例如：'层数默认为 1', '基底默认为 Suspended'。如果无缺省，写'无']

        #### 4. ⚖️ 物理机制解读
        [这里请生成一段 50-100 字的学术分析。
        逻辑模板：当前温度为 [T]K，主要受 [Umklapp/缺陷] 散射机制主导。样品长度 [L]um 处于 [弹道/扩散] 输运区间。ML 预测值 [符合/偏离] 理论预期...]
        ---
        """),
        
        # 历史对话记忆
        MessagesPlaceholder(variable_name="chat_history"),
        
        ("human", "{input}"),
        
        # 预留给 Agent 思考的暂存区 (它会在这里疯狂思考 Step 1/2/3，但用户看不见)
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    # 4. 绑定工具
    agent = create_tool_calling_agent(llm, tools, prompt)

    # 5. 创建记忆模块
    memory = ConversationBufferMemory(
        memory_key="chat_history", 
        return_messages=True
    )

    # 6. 创建执行器
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True,
        memory=memory,
        max_iterations=8,             # 允许它多想几步
        handle_parsing_errors=True,   # 容错
        early_stopping_method="generate"
    )
    

    return agent_executor
