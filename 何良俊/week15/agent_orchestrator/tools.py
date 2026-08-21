"""
tools — agent 可注册的通用小工具
=================================

工具保持"小而完整"：calculator 用于演示工具调用闭环（模型决策 → 执行 → 回传观察），
current_date 提供常识性上下文。subagent 与 orchestrator 共用此注册表，
新增工具只需实现一个普通函数并声明 schema，无需改动 agent 循环。
"""
from __future__ import annotations

import ast
import datetime
import math
import operator


def current_date() -> str:
    """返回今天日期（YYYY-MM-DD）。"""
    return datetime.date.today().isoformat()


# ── calculator：受限的安全表达式计算 ──────────────────────────────────────────

_ALLOWED_BINOPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod, ast.Pow: operator.pow,
}
_ALLOWED_FUNCS = {
    "abs": abs, "round": round, "min": min, "max": max,
    "sqrt": math.sqrt, "pow": pow, "floor": math.floor, "ceil": math.ceil,
}
_CONSTANTS = {"pi": math.pi, "e": math.e}


def _check_node(node: ast.AST, depth: int = 0) -> None:
    """AST 白名单校验：只允许数值常量、算术运算、白名单函数与常量标识符。"""
    if depth > 8:
        raise ValueError("表达式嵌套过深")
    if isinstance(node, ast.Expression):
        _check_node(node.body, depth + 1)
    elif isinstance(node, ast.Constant):
        if not isinstance(node.value, (int, float)):
            raise ValueError("只允许数值常量")
    elif isinstance(node, ast.BinOp):
        if type(node.op) not in _ALLOWED_BINOPS:
            raise ValueError(f"不支持的运算符 {type(node.op).__name__}")
        _check_node(node.left, depth + 1)
        _check_node(node.right, depth + 1)
    elif isinstance(node, ast.UnaryOp):
        if not isinstance(node.op, (ast.USub, ast.UAdd)):
            raise ValueError("只支持一元正负号")
        _check_node(node.operand, depth + 1)
    elif isinstance(node, ast.Name):
        if node.id not in _CONSTANTS:
            raise ValueError(f"不允许的标识符 {node.id!r}")
    elif isinstance(node, ast.Call):
        if node.keywords:
            raise ValueError("不允许关键字参数")
        fn = node.func.id if isinstance(node.func, ast.Name) else None
        if fn not in _ALLOWED_FUNCS:
            raise ValueError(f"不允许的函数 {fn!r}")
        for a in node.args:
            _check_node(a, depth + 1)
    else:
        raise ValueError(f"不允许的表达式节点 {type(node).__name__}")


def calculator(expression: str) -> str:
    """安全计算数学表达式，返回计算结果字符串。

    支持: + - * / // % ** 括号、一元正负号、常量 pi/e，
    函数 abs/round/min/max/sqrt/pow/floor/ceil。
    任何解析/计算失败都返回可读错误字符串，而不是抛出异常。
    """
    expression = (expression or "").strip()
    if not expression:
        return "错误: 表达式为空"
    try:
        tree = ast.parse(expression, mode="eval")
        _check_node(tree)
        value = eval(  # noqa: S307 — 已通过上方 AST 白名单校验
            compile(tree, "<expr>", "eval"),
            {"__builtins__": {}},
            {**_CONSTANTS, **_ALLOWED_FUNCS},
        )
        return str(value)
    except Exception as e:
        return f"计算失败: {e}"
