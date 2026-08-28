# -*- coding: utf-8 -*-
"""桩掉 openai/src 后导入真实 run_function_call，用假 client 验证多轮循环逻辑。
不联网、不发 API。验证四条路径：天气多轮(两步) / 无工具直答 / 触顶强制收尾 / 并行两步对比。"""
import sys, types, json

# ── 桩依赖 ────────────────────────────────────────────────────────────────
openai_mod = types.ModuleType("openai")
class _OpenAI:  # 不会被实例化（run 接收外部传入的 client）
    pass
openai_mod.OpenAI = _OpenAI
sys.modules["openai"] = openai_mod

src_pkg = types.ModuleType("src"); sys.modules["src"] = src_pkg
w_mod = types.ModuleType("src.weather_backend")
# 两步工具的 mock：geocode 返回 dict（FC loop 会 JSON 序列化成字符串），get_weather_by_coords 返回字符串
w_mod.geocode = lambda city: {"lat": 1.0, "lon": 2.0, "name": city, "country": "C", "admin1": "A"}
w_mod.get_weather_by_coords = lambda **kw: f"WEATHER({kw.get('name')})"
sys.modules["src.weather_backend"] = w_mod

sys.path.insert(0, "/Users/zhanglei/projects/hub-TroE/张雷/week11")
import mode_function_call.run_function_call as m  # noqa: E402

# ── 假对象 ────────────────────────────────────────────────────────────────
class Fn:
    def __init__(self, name, args): self.name = name; self.arguments = json.dumps(args)
class TC:
    def __init__(self, name, args, cid="c1"): self.id = cid; self.function = Fn(name, args)
class Msg:
    def __init__(self, tool_calls=None, content=None):
        self.tool_calls = tool_calls; self.content = content
class Choice:
    def __init__(self, msg): self.message = msg
class Resp:
    def __init__(self, msg): self.choices = [Choice(msg)]

class FakeClient:
    def __init__(self, resps): self._resps = list(resps); self.calls = []
    @property
    def chat(self): return self
    @property
    def completions(self): return self
    def create(self, **kw):
        self.calls.append(kw)
        return self._resps.pop(0)

def run_case(name, resps, expect_rounds, expect_tools, expect_answer_contains=None,
             expect_last_tool_choice=None):
    fc = FakeClient(resps)
    r = m.run(fc, "fake-model", "q", verbose=False)
    ok_rounds = r["rounds"] == expect_rounds
    ok_tools = [t["name"] for t in r["tool_calls"]] == expect_tools
    ok_ans = (expect_answer_contains in (r["answer"] or "")) if expect_answer_contains else True
    ok_tc = (fc.calls[-1].get("tool_choice") == expect_last_tool_choice) if expect_last_tool_choice else True
    status = "PASS" if (ok_rounds and ok_tools and ok_ans and ok_tc) else "FAIL"
    print(f"[{status}] {name}: rounds={r['rounds']}(want {expect_rounds}) "
          f"tools={[t['name'] for t in r['tool_calls']]}(want {expect_tools}) "
          f"ans={r['answer'][:30]!r} last_tc={fc.calls[-1].get('tool_choice')}")
    if status == "FAIL":
        print("    tool_call args:", [t['args'] for t in r['tool_calls']])
        print("    full answer:", r['answer'])
    return status == "PASS"

results = []

# 坐标参数模板（模拟 geocode 返回、再由模型转发给 get_weather_by_coords）
COORDS = {"lat": 1.0, "lon": 2.0, "country": "C", "admin1": "A"}

# 1) 天气多轮(两步)：geocode(宁德)->weather(宁德)->见气温->geocode(哈尔滨)->weather(哈尔滨)->答
#    每个城市 = geocode + get_weather_by_coords 两步，坐标数据依赖强制多轮 => 5 次请求，4 次工具调用
results.append(run_case(
    "天气多轮两步",
    [Resp(Msg(tool_calls=[TC("geocode", {"city": "宁德"})])),
     Resp(Msg(tool_calls=[TC("get_weather_by_coords", {**COORDS, "name": "宁德"})])),
     Resp(Msg(tool_calls=[TC("geocode", {"city": "哈尔滨"})])),
     Resp(Msg(tool_calls=[TC("get_weather_by_coords", {**COORDS, "name": "哈尔滨"})])),
     Resp(Msg(content="宁德和哈尔滨天气对比完毕"))],
    expect_rounds=5,
    expect_tools=["geocode", "get_weather_by_coords", "geocode", "get_weather_by_coords"],
    expect_answer_contains="对比完毕"))

# 2) 无工具直答：模型第一轮就给答案 => rounds=1，0 次工具调用
results.append(run_case(
    "无工具直答",
    [Resp(Msg(content="直接回答"))],
    expect_rounds=1, expect_tools=[], expect_answer_contains="直接回答"))

# 3) 触顶强制收尾：连续 8 轮都调 geocode => 第 9 次请求 tool_choice=none
results.append(run_case(
    "触顶收尾",
    [Resp(Msg(tool_calls=[TC("geocode", {"city": "宁德"})]))] * 8
     + [Resp(Msg(content="强制收尾答案"))],
    expect_rounds=8, expect_tools=["geocode"] * 8,
    expect_answer_contains="强制收尾", expect_last_tool_choice="none"))

# 4) 并行两步对比：R1 同时 geocode(北京+上海) -> R2 同时 weather(北京+上海) -> R3 答
#    验证单轮多工具调用 + 两步数据依赖（weather 必须等 geocode 出坐标才能调）
results.append(run_case(
    "并行两步对比",
    [Resp(Msg(tool_calls=[TC("geocode", {"city": "北京"}, cid="c1"),
                          TC("geocode", {"city": "上海"}, cid="c2")])),
     Resp(Msg(tool_calls=[TC("get_weather_by_coords", {**COORDS, "name": "北京"}, cid="c3"),
                          TC("get_weather_by_coords", {**COORDS, "name": "上海"}, cid="c4")])),
     Resp(Msg(content="北京和上海对比完毕"))],
    expect_rounds=3,
    expect_tools=["geocode", "geocode", "get_weather_by_coords", "get_weather_by_coords"],
    expect_answer_contains="对比完毕"))

print("\n==== 全部通过" if all(results) else "\n==== 存在失败", "====")
sys.exit(0 if all(results) else 1)
