# agent.py（扩展版）

import os
import argparse
import json
from openai import OpenAI
from tools import TOOLS_MAP, TOOLS_SCHEMA

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# ========== 配置（支持 DeepSeek / DashScope） ==========
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)
MODEL = os.getenv("AGENT_MODEL", "deepseek-v4-flash")

# 若使用 DashScope，取消下面注释并注释掉上面
# client = OpenAI(
#     api_key=os.getenv("DASHSCOPE_API_KEY"),
#     base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
# )
# MODEL = os.getenv("AGENT_MODEL", "qwen-max")

SYSTEM_PROMPT = """你是一个专业的A股金融分析助手。
规则：
- 调用 financial_indicator 或 stock_price 之前，必须先用 company_lookup 获取股票代码
- 数字计算必须使用 calculator 工具，不能心算
- Final Answer 必须引用具体数据来源
- 如果没有合适工具能回答，直接说明原因
- 多轮对话中请结合历史上下文，保持回答一致性
"""

# ========== 多轮对话 Agent 类 ==========
class ChatAgent:
    def __init__(self):
        self.messages = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]
        self.tools = TOOLS_SCHEMA
        self.max_steps = 10

    def ask(self, user_question: str, stream=False):
        """
        处理一轮用户提问，返回最终答案（或流式输出每步）。
        若 stream=True，则 yield 每一步的字典；否则直接返回最终答案字符串。
        """
        self.messages.append({"role": "user", "content": user_question})

        for step in range(1, self.max_steps + 1):
            response = client.chat.completions.create(
                model=MODEL,
                messages=self.messages,
                tools=self.tools,
                tool_choice="auto",
                temperature=0,
            )
            msg = response.choices[0].message
            reason = response.choices[0].finish_reason

            # 模型直接回答
            if reason == "stop" or not msg.tool_calls:
                self.messages.append(msg)  # 保存助手回复
                final_answer = msg.content or "（模型返回空）"
                if stream:
                    yield {"type": "final", "answer": final_answer}
                else:
                    return final_answer

            # 模型请求工具调用
            self.messages.append(msg)  # 保存带有 tool_calls 的助手消息
            for tool_call in msg.tool_calls:
                tool_name = tool_call.function.name
                try:
                    tool_args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    tool_args = {}

                tool_fn = TOOLS_MAP.get(tool_name)
                if tool_fn is None:
                    observation = f"未知工具 '{tool_name}'"
                else:
                    try:
                        observation = tool_fn(**tool_args)
                    except Exception as e:
                        observation = f"工具执行失败: {e}"

                # 将工具结果以 role=tool 追加到消息历史
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(observation),
                })

                if stream:
                    yield {
                        "type": "action",
                        "step": step,
                        "action": tool_name,
                        "action_input": tool_args,
                        "observation": str(observation)[:300],
                    }

        # 达到最大步数
        error_msg = f"已达最大步数 {self.max_steps}，未能得出最终答案。"
        if stream:
            yield {"type": "error", "message": error_msg}
        return error_msg

    def reset(self):
        """重置会话（清空历史，仅保留 System Prompt）"""
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

# ========== 命令行交互 ==========
def interactive_chat():
    agent = ChatAgent()
    print("\n🤖 多轮对话 Agent 已启动（输入 'exit' 退出，'reset' 重置会话）\n")
    while True:
        user_input = input("你: ").strip()
        if user_input.lower() in ("exit", "quit"):
            print("再见！")
            break
        if user_input.lower() == "reset":
            agent.reset()
            print("✅ 会话已重置\n")
            continue
        if not user_input:
            continue

        print("\n🧠 思考中...")
        # 流式打印每步（可选）
        for step_data in agent.ask(user_input, stream=True):
            if step_data["type"] == "action":
                print(f"  [Step {step_data['step']}] 🔧 {step_data['action']} -> {step_data['observation'][:60]}...")
            elif step_data["type"] == "final":
                print(f"\n🤖 最终回答:\n{step_data['answer']}\n")
            elif step_data["type"] == "error":
                print(f"⚠️  {step_data['message']}\n")

# ========== 原有单次运行保持不变（兼容） ==========
def run_single_question(question, max_steps=10, mode="fc"):
    if mode == "manual":
        from react_manual import run_and_print
    else:
        from react_function_calling import run_and_print
    run_and_print(question, max_steps)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--chat", action="store_true", help="启动多轮对话模式")
    parser.add_argument("--question", default=None, help="单次问答的问题")
    parser.add_argument("--max_steps", type=int, default=10)
    parser.add_argument("--mode", choices=["manual", "fc"], default="fc")
    args = parser.parse_args()

    if args.chat:
        interactive_chat()
    elif args.question:
        run_single_question(args.question, args.max_steps, args.mode)
    else:
        # 默认单次问答（保留原有行为）
        run_single_question("贵州茅台和五粮液2023年的毛利率哪家更高？差多少个百分点？", args.max_steps, args.mode)
