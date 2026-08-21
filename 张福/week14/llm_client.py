# -*- coding: utf-8 -*-
"""
LLM 客户端：基于 DASHSCOPE_API_KEY 调用阿里云百炼 (DashScope) 的通义千问模型。
使用 OpenAI 兼容接口，无需额外 SDK，仅依赖 requests。
"""
import os
import json
import requests

# DashScope OpenAI 兼容接口
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
DEFAULT_MODEL = "qwen-plus"


def get_api_key():
    """从环境变量获取 DASHSCOPE_API_KEY"""
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "未检测到 DASHSCOPE_API_KEY 环境变量，请先设置：\n"
            "  export DASHSCOPE_API_KEY=\"你的密钥\""
        )
    return api_key


# PMP 专家系统提示词
PMP_SYSTEM_PROMPT = (
    "你是一位资深 PMP（项目管理专业人士）专家，精通 PMBOK 指南的全部知识体系，"
    "包括项目整体管理、范围管理、进度管理、成本管理、质量管理、资源管理、"
    "沟通管理、采购管理、风险管理、相关方管理十大知识领域。\n\n"
    "你的职责：\n"
    "1. 针对用户提出的项目管理问题，给出专业、准确、具有指导性的意见。\n"
    "2. 回答应结合 PMBOK 的过程、工具与技术、输入输出，体现专业性。\n"
    "3. 当用户提出与 PMP（项目管理）无关的知识点时，你必须且只能回复：\n"
    "   \"很抱歉，当前并不能回答您PMP以外的知识领域知识。\"\n"
    "4. 判断是否属于 PMP 范畴的标准：问题是否涉及项目管理、PMBOK 知识领域、"
    "项目管理过程、项目管理工具技术、项目经理职责等内容。"
)


def _build_system_prompt(skill_content=None):
    """
    构建系统提示词。当有匹配的 skill 上下文时，注入该知识领域的
    标准问答内容，引导 LLM 基于标准体系回答。

    参数:
        skill_content: 匹配到的 skill Markdown 正文，None 则返回默认提示词

    返回:
        str: 完整的系统提示词
    """
    if not skill_content:
        return PMP_SYSTEM_PROMPT

    # 有 skill 上下文时，在系统提示词后追加知识领域标准问答
    return (
        PMP_SYSTEM_PROMPT
        + "\n\n"
        + "=== 当前问题所属知识领域的标准参考内容 ===\n"
        + "请优先参考以下标准问答内容来回答用户的问题，确保回答与标准体系一致：\n\n"
        + skill_content
        + "\n\n"
        + "=== 参考内容结束 ===\n"
        + "回答要求：\n"
        + "1. 严格基于上述标准参考内容作答，确保专业性和准确性。\n"
        + "2. 如参考内容中有直接匹配的标准答案，请以其为核心进行回答，可适当展开。\n"
        + "3. 保持与PMBOK指南的一致性，使用标准术语。"
    )


def chat(question, history=None, model=None, temperature=0.7, skill_content=None):
    """
    调用通义千问模型进行对话。

    参数:
        question: 用户问题
        history: 历史对话列表 [{"role": "user/assistant", "content": "..."}]
        model: 模型名称，默认 qwen-plus
        temperature: 生成温度
        skill_content: 匹配到的 skill Markdown 正文，None 则使用默认系统提示词

    返回:
        dict: {"answer": "模型回答", "model": "模型名", "success": True/False, "error": "..."}
    """
    api_key = get_api_key()
    model = model or DEFAULT_MODEL

    messages = [{"role": "system", "content": _build_system_prompt(skill_content)}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": question})

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }

    try:
        resp = requests.post(
            DASHSCOPE_BASE_URL,
            headers=headers,
            json=payload,
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        answer = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)
        return {
            "answer": answer,
            "model": model,
            "success": True,
            "error": None,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }
    except requests.exceptions.HTTPError as e:
        err_body = ""
        try:
            err_body = resp.text
        except Exception:
            pass
        return {
            "answer": None,
            "model": model,
            "success": False,
            "error": f"HTTP 错误: {e} | 响应: {err_body[:300]}",
        }
    except requests.exceptions.RequestException as e:
        return {
            "answer": None,
            "model": model,
            "success": False,
            "error": f"请求异常: {e}",
        }
    except (KeyError, IndexError) as e:
        return {
            "answer": None,
            "model": model,
            "success": False,
            "error": f"解析响应失败: {e} | 原始: {resp.text[:300]}",
        }


if __name__ == "__main__":
    # 简单自测
    result = chat("什么是项目章程？")
    print("成功:", result["success"])
    print("回答:", result.get("answer"))
    if result["error"]:
        print("错误:", result["error"])
