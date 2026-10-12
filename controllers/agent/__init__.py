"""大模型 agent 当国家的战略层（通用：简报、可选动作都从当前剧本的规则集生成，不写死二战）。

- directive.py：规则 AI 读取 agent 方针的软权重（战线目标、攻守、造兵偏好）；
- brief.py：给大模型看的精简局势简报；
- llm.py：OpenAI 兼容的大模型客户端（默认 DeepSeek，密钥从环境变量或 ~/.config/sekai/secrets.json 读）；
- runner.py：判断哪些 agent 国家这回合该决策（每 N 回合 + 大事触发），调用大模型，把回答转成指令。

大模型调用在引擎 step 之外（server 推进回合之前）：step 保持确定性，agent 的决定以指令形式进入回合，
可以回放。
"""
