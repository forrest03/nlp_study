"""主 Agent 和子 Agent 共用的 ReAct 循环。"""

import re
import time

from src.llm_client import llm_chat


REACT_SYSTEM = """你是通用任务执行助手。
可用工具：
{tools}

每一轮严格输出以下格式：
Thought: 简短说明下一步推理
Action: 工具名
Action Input: 工具参数

工具执行后会给你 Observation。信息足够时，严格输出：
Thought: 我已收集足够信息
Final Answer: 最终回答，按任务需要组织内容，并保留来源。
每轮只能调用一个工具。"""


class ReActLoop:
    """工具集决定 Agent 的能力边界。"""

    def __init__(self, name, tools, max_steps, system_prompt=None):
        self.name = name
        self.tools = tools
        self.max_steps = max_steps
        self.system_prompt = system_prompt or REACT_SYSTEM

    def run(self, question, on_step=None, shared_state=None):
        start_time = time.time()
        trace = []
        tools_text = "\n".join("- " + name + ": " + item[1] for name, item in self.tools.items())
        system = self.system_prompt.format(tools=tools_text)
        history = "Question: " + question + "\n\n"

        for index in range(self.max_steps):
            output = llm_chat(system, history, stop=["Observation:"])
            thought, action, action_input = self.parse_output(output)
            step = {
                "idx": index,
                "agent": self.name,
                "thought": thought,
                "action": action,
                "action_input": action_input,
                "observation": "",
            }

            if action == "Final Answer":
                step["final"] = True
                trace.append(step)
                if on_step:
                    on_step(step)
                return self.build_result(action_input, trace, start_time)

            observation = self.execute_tool(action, action_input, shared_state)
            step["observation"] = observation
            trace.append(step)
            if on_step:
                on_step(step)
            history += output + "\nObservation: " + observation[:1500] + "\n"

        answer = "达到最大步骤，最后一次工具结果如下：\n" + trace[-1]["observation"]
        return self.build_result(answer, trace, start_time)

    def build_result(self, answer, trace, start_time):
        return {
            "final_answer": answer,
            "trace": trace,
            "duration": round(time.time() - start_time, 2),
        }

    def parse_output(self, output):
        """解析 LLM 的 Thought / Action；无格式的实质文本作为最终回答兜底。"""
        thought_match = re.search(r"Thought:\s*(.*?)(?=\nAction:|\nFinal Answer:|$)", output, re.S)
        thought = thought_match.group(1).strip() if thought_match else ""
        final_match = re.search(r"Final Answer:\s*(.*)", output, re.S)
        if final_match:
            return thought, "Final Answer", final_match.group(1).strip()
        action_match = re.search(r"Action:\s*(.*)", output)
        input_match = re.search(r"Action Input:\s*(.*)", output)
        if action_match:
            action_input = input_match.group(1).strip() if input_match else ""
            return thought, action_match.group(1).strip(), action_input
        return thought or "模型直接给出了回答", "Final Answer", output.strip()

    def execute_tool(self, action, action_input, shared_state):
        if action not in self.tools:
            return "工具不存在: " + action + "；可用工具: " + ", ".join(self.tools.keys())
        function = self.tools[action][0]
        try:
            return str(function(action_input, shared_state=shared_state))
        except Exception as error:
            return "工具执行错误: " + type(error).__name__ + ": " + str(error)[:160]
