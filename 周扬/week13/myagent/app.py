'''
    使用纯python搭建一个用户交互界面：
    # 用户输入对话并回车
    # 智能体的回复
    # 智能体的中间过程，比如如何调用了技能、过程中如何思考的，要展示打印出来

    支持的命令：
    - /model  : 进入模型配置页面
    - /status : 查看当前模型配置
    - /clear  : 清屏
    - /help   : 显示帮助信息
    - quit    : 退出程序
'''

from dataclasses import dataclass, field
from typing import Generator, Dict, Any, Optional

from model import get_global_config, interactive_config, show_current_config
from agent_core import AgentCore
from skill_manager import SkillManager
from tool_manager import ToolManager
from memery_manager import MemoryManager


SESSION_ID = "cli_session"
memory_manager = MemoryManager()
tool_manager = ToolManager(memory_manager=memory_manager)
skill_manager = SkillManager()
agent_core = AgentCore(
    tool_manager=tool_manager,
    skill_manager=skill_manager,
    memory_manager=memory_manager,
    session_id=SESSION_ID,
)


# ANSI 颜色代码
class Colors:
    RESET = '\033[0m'
    BOLD = '\033[1m'
    
    # 用户输入
    USER = '\033[36m'      # 青色
    
    # 智能体回复
    AGENT = '\033[32m'     # 绿色
    
    # 中间过程
    THINKING = '\033[33m'  # 黄色
    TOOL_CALL = '\033[35m' # 紫色
    TOOL_RESULT = '\033[34m' # 蓝色
    SKILL = '\033[94m'     # 亮蓝
    CONTEXT = '\033[96m'   # 亮青
    
    # 错误
    ERROR = '\033[31m'     # 红色
    
    # 分隔线
    SEPARATOR = '\033[90m' # 灰色
    
    # 系统命令
    COMMAND = '\033[36m'   # 青色


@dataclass
class TraceStats:
    """记录一轮对话的执行轨迹摘要。"""
    steps: int = 0
    thinking_events: int = 0
    tool_calls: int = 0
    tool_success: int = 0
    tool_failed: int = 0
    matched_skills: list[str] = field(default_factory=list)
    new_skills: list[str] = field(default_factory=list)


def print_welcome():
    """打印欢迎信息"""
    print(f"\n{Colors.BOLD}{'='*60}{Colors.RESET}")
    print(f"{Colors.BOLD}  智能体交互界面{Colors.RESET}")
    print(f"{Colors.BOLD}{'='*60}{Colors.RESET}")
    print(f"  输入你的问题，按回车发送")
    print(f"  输入 {Colors.BOLD}'/help'{Colors.RESET} 查看所有命令")
    print(f"{Colors.BOLD}{'='*60}{Colors.RESET}\n")


def print_help():
    """打印帮助信息"""
    print(f"\n{Colors.BOLD}可用命令：{Colors.RESET}")
    print(f"  {Colors.COMMAND}/model{Colors.RESET}   - 配置模型（Base URL / API Key / 选择模型）")
    print(f"  {Colors.COMMAND}/status{Colors.RESET}  - 查看当前模型配置")
    print(f"  {Colors.COMMAND}/memory{Colors.RESET}  - 查看当前记忆摘要")
    print(f"  {Colors.COMMAND}/skills{Colors.RESET}  - 查看当前 skill 清单")
    print(f"  {Colors.COMMAND}/reset{Colors.RESET}   - 重置已加载的 skill 状态")
    print(f"  {Colors.COMMAND}/clear{Colors.RESET}   - 清屏")
    print(f"  {Colors.COMMAND}/help{Colors.RESET}    - 显示此帮助信息")
    print(f"  {Colors.COMMAND}quit{Colors.RESET}     - 退出程序")
    print()


def print_separator():
    """打印分隔线"""
    print(f"{Colors.SEPARATOR}{'─'*60}{Colors.RESET}")


def print_trace_header():
    """打印执行轨迹标题"""
    print(f"{Colors.BOLD}{Colors.SEPARATOR}执行轨迹{Colors.RESET}")
    print_separator()


def print_user_input(text: str):
    """打印用户输入"""
    print(f"\n{Colors.USER}{Colors.BOLD}👤 你：{Colors.RESET}{Colors.USER}{text}{Colors.RESET}")


def print_agent_response(text: str):
    """打印智能体最终回复"""
    print(f"\n{Colors.AGENT}{Colors.BOLD}🤖 智能体：{Colors.RESET}")
    print(f"{Colors.AGENT}{text}{Colors.RESET}")


def print_thinking(text: str):
    """打印思考过程"""
    print(f"  {Colors.THINKING}💭 思考：{text}{Colors.RESET}")


def print_tool_call(tool_name: str, arguments: Dict[str, Any]):
    """打印工具调用"""
    print(f"  {Colors.TOOL_CALL}🔧 调用工具：{tool_name}{Colors.RESET}")
    if arguments:
        for key, value in arguments.items():
            print(f"     {Colors.TOOL_CALL}├─ {key}: {value}{Colors.RESET}")


def print_tool_result(result: Any, success: bool = True):
    """打印工具执行结果"""
    icon = "✅" if success else "❌"
    color = Colors.TOOL_RESULT if success else Colors.ERROR
    lines = str(result).splitlines() or [str(result)]
    print(f"     {color}{icon} 结果：{lines[0]}{Colors.RESET}")
    for extra_line in lines[1:]:
        print(f"     {color}   {extra_line}{Colors.RESET}")


def print_error(text: str):
    """打印错误信息"""
    print(f"\n{Colors.ERROR}❌ 错误：{text}{Colors.RESET}")


def print_step(step_num: int, description: str):
    """打印步骤信息"""
    print(f"  {Colors.THINKING}📍 步骤 {step_num}：{description}{Colors.RESET}")


def print_context_status(session_message_count: int, session_summary_exists: bool, loaded_skill_count: int):
    """打印会话上下文状态。"""
    summary_text = "有" if session_summary_exists else "无"
    print(
        f"  {Colors.CONTEXT}🧭 上下文："
        f"历史消息 {session_message_count} 条，"
        f"session 摘要 {summary_text}，"
        f"已加载 skill {loaded_skill_count} 个{Colors.RESET}"
    )


def print_skill_status(matched_skills: list[str], new_skills: list[str], loaded_skills: list[str]):
    """打印 skill 命中与加载状态。"""
    matched_text = "、".join(matched_skills) if matched_skills else "未命中"
    new_text = "、".join(new_skills) if new_skills else "无新加载"
    loaded_text = "、".join(loaded_skills) if loaded_skills else "暂无"
    print(f"  {Colors.SKILL}🧩 Skill 命中：{matched_text}{Colors.RESET}")
    print(f"  {Colors.SKILL}   新加载：{new_text}{Colors.RESET}")
    print(f"  {Colors.SKILL}   当前已加载：{loaded_text}{Colors.RESET}")


def print_model_decision(decision: str, tool_calls: list[str]):
    """打印模型本轮决策。"""
    if decision == "tool_call":
        tools_text = "、".join(tool_calls) if tool_calls else "无"
        print(f"  {Colors.THINKING}🧠 决策：继续调用工具 -> {tools_text}{Colors.RESET}")
    else:
        print(f"  {Colors.THINKING}🧠 决策：当前信息已足够，直接生成最终回答{Colors.RESET}")


def print_trace_summary(stats: TraceStats):
    """打印一轮对话结束后的轨迹摘要。"""
    matched_text = "、".join(stats.matched_skills) if stats.matched_skills else "无"
    new_text = "、".join(stats.new_skills) if stats.new_skills else "无"
    print(f"\n{Colors.BOLD}{Colors.SEPARATOR}轨迹摘要{Colors.RESET}")
    print(f"  步骤数：{stats.steps}")
    print(f"  思考事件：{stats.thinking_events}")
    print(f"  工具调用：{stats.tool_calls}（成功 {stats.tool_success} / 失败 {stats.tool_failed}）")
    print(f"  命中 skill：{matched_text}")
    print(f"  新加载 skill：{new_text}")


def clear_screen():
    """清屏"""
    print("\033c", end="")


def get_user_input() -> Optional[str]:
    """获取用户输入"""
    try:
        user_input = input(f"\n{Colors.USER}{Colors.BOLD}>>> {Colors.RESET}").strip()
        return user_input if user_input else None
    except (EOFError, KeyboardInterrupt):
        print("\n")
        return "quit"


def real_agent_response(user_input: str) -> Generator[Dict[str, Any], None, None]:
    """
    真实的智能体响应：调用已配置的 LLM 模型
    
    参数：
        user_input: 用户输入的文本
    
    yield 事件字典，界面根据 type 渲染
    """
    config = get_global_config()
    yield from agent_core.run(user_input, config=config, max_steps=8)


def simulate_agent_response(user_input: str) -> Generator[Dict[str, Any], None, None]:
    """
    模拟智能体响应流（未配置模型时使用）
    """
    yield from agent_core.simulate(user_input)


def handle_event(event: Dict[str, Any], stats: Optional[TraceStats] = None):
    """处理单个事件"""
    event_type = event.get("type")
    
    if event_type == "thinking":
        if stats is not None:
            stats.thinking_events += 1
        print_thinking(event["content"])
    
    elif event_type == "tool_call":
        if stats is not None:
            stats.tool_calls += 1
        print_tool_call(event["tool"], event.get("args", {}))
    
    elif event_type == "tool_result":
        if stats is not None:
            if event.get("success", True):
                stats.tool_success += 1
            else:
                stats.tool_failed += 1
        print_tool_result(event["result"], event.get("success", True))
    
    elif event_type == "step":
        if stats is not None:
            stats.steps += 1
        print_step(event["num"], event["desc"])
    
    elif event_type == "context_status":
        print_context_status(
            session_message_count=event.get("session_message_count", 0),
            session_summary_exists=event.get("session_summary_exists", False),
            loaded_skill_count=event.get("loaded_skill_count", 0),
        )

    elif event_type == "skill_status":
        matched_skills = event.get("matched_skills", [])
        new_skills = event.get("new_skills", [])
        if stats is not None:
            for item in matched_skills:
                if item not in stats.matched_skills:
                    stats.matched_skills.append(item)
            for item in new_skills:
                if item not in stats.new_skills:
                    stats.new_skills.append(item)
        print_skill_status(
            matched_skills=matched_skills,
            new_skills=new_skills,
            loaded_skills=event.get("loaded_skills", []),
        )

    elif event_type == "model_decision":
        print_model_decision(
            decision=event.get("decision", ""),
            tool_calls=event.get("tool_calls", []),
        )

    elif event_type == "response":
        print_agent_response(event["content"])
    
    elif event_type == "memory_write":
        print_thinking(f"记忆写入 -> {event['target']}: {event['content']}")

    elif event_type == "error":
        print_error(event["content"])


def process_user_input(user_input: str):
    """
    处理用户输入
    
    如果已配置模型，使用真实 LLM；否则使用模拟响应
    """
    print_user_input(user_input)
    print_separator()
    print_trace_header()
    
    config = get_global_config()
    stats = TraceStats()
    
    # 根据是否配置模型选择响应方式
    if config.is_configured():
        agent_func = real_agent_response
    else:
        agent_func = simulate_agent_response
    
    try:
        for event in agent_func(user_input):
            handle_event(event, stats=stats)
    except Exception as e:
        print_error(f"处理过程中出现错误：{str(e)}")
    
    print_trace_summary(stats)
    print_separator()


def show_memory_status():
    """打印记忆状态。"""
    summary = agent_core.get_memory_status()
    print("\n当前记忆状态：")
    print(f"  soul: {summary['soul_file']}")
    print(f"  user: {summary['user_file']}")
    print(f"  memery: {summary['memery_file']}")
    print(f"  session file: {summary['session_file']}")
    print(f"  session summary file: {summary['session_summary_file']}")
    print(f"  session summary exists: {summary['session_summary_exists']}")
    print(f"  session message count: {summary['session_message_count']}")
    print(f"  backend: {summary['memory_backend']}")


def show_skill_status():
    """打印 skill 状态。"""
    status = agent_core.get_skill_status()
    print("\n当前 skill 清单：")
    print(status["manifest_text"])
    print("\n已加载的 skill：")
    if status["loaded_skills"]:
        print("  " + "、".join(status["loaded_skills"]))
    else:
        print("  暂无")


def main():
    """主循环"""
    clear_screen()
    print_welcome()
    
    # 启动时检查模型配置
    config = get_global_config()
    if config.is_configured():
        print(f"{Colors.AGENT}✓ 已加载模型配置: {config.model_name}{Colors.RESET}")
    else:
        print(f"{Colors.THINKING}⚠ 未配置模型，请输入 /model 进行配置{Colors.RESET}")
    
    while True:
        user_input = get_user_input()
        
        if user_input is None:
            continue
        
        # 处理特殊命令
        cmd = user_input.lower()
        
        if cmd in ["quit", "exit", "q"]:
            print(f"\n{Colors.AGENT}再见！{Colors.RESET}\n")
            break
        
        elif cmd == "/model":
            interactive_config()
        
        elif cmd == "/status":
            show_current_config()

        elif cmd == "/memory":
            show_memory_status()

        elif cmd == "/skills":
            show_skill_status()

        elif cmd == "/reset":
            agent_core.reset_session()
            print(f"{Colors.AGENT}✓ 已重置会话状态和渐进式加载状态{Colors.RESET}")
        
        elif cmd == "/clear":
            clear_screen()
            print_welcome()
        
        elif cmd == "/help":
            print_help()
        
        else:
            # 普通对话输入
            process_user_input(user_input)


if __name__ == "__main__":
    main()
