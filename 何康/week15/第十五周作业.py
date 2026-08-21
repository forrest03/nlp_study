grqph.py：实现任务图
@dataclass
class TaskNode:
    name: str
    task: str
    deps: list[str] = field(default_factory=list)
    result: Any = None

class TaskGraph:
    def __init__(self):
        self.nodes = {}

    def add_task(self, name, task, deps=None):
        self.nodes[name] = TaskNode(
            name=name,
            task=task,
            deps=deps or []
        )

    def ready_tasks(self, finished):
        return [
            node for node in self.nodes.values()
            if node.name not in finished
            and all(dep in finished for dep in node.deps)
        ]

    def is_finished(self, finished):
        return len(finished) == len(self.nodes)


subagent.py:负责一个独立任务
class LLM:
    """简单的 LLM 抽象层，后续可以替换成 OpenAI/Qwen 等模型。"""

    def generate(self, prompt):
        time.sleep(random.uniform(0.5, 1.5))

        return f"LLM分析结果：{prompt}"
class SubAgent:
    def __init__(self, name, role, llm=None):
        self.name = name
        self.role = role
        self.llm = llm or LLM()

    def run(self, task):
        prompt = f"""
你是一个{self.role}。

请完成下面的任务：
{task}

要求：
1. 独立分析
2. 给出关键结论
3. 输出简洁
"""
       result = self.llm.generate(prompt)

        return {
            "agent": self.name,
            "role": self.role,
            "task": task,
            "result": result
        }


agent.py:核心代码agent
class Agent:
    def __init__(self, max_workers=4):
        self.max_workers = max_workers

        self.subagents = {
            "research": SubAgent(
                "ResearchAgent",
                "资料分析专家"
            ),
            "technical": SubAgent(
                "TechnicalAgent",
                "技术分析专家"
            ),
            "creative": SubAgent(
                "CreativeAgent",
                "方案设计专家"
            ),
            "summary": SubAgent(
                "SummaryAgent",
                "总结专家"
            )
        }

    def build_graph(self, user_task):
        graph = TaskGraph()

        graph.add_task(
            "research",
            f"分析以下任务的背景和相关信息：{user_task}"
        )

        graph.add_task(
            "technical",
            f"从技术角度分析以下任务：{user_task}"
        )

        graph.add_task(
            "creative",
            f"设计解决以下任务的方案：{user_task}"
        )

        graph.add_task(
            "summary",
            f"根据前面多个Agent的分析结果，对任务进行最终总结：{user_task}",
            deps=["research", "technical", "creative"]
        )

        return graph

    def execute(self, user_task):
        graph = self.build_graph(user_task)

        finished = set()
        results = {}

        print("=" * 60)
        print("Main Agent 开始执行")
        print(f"任务：{user_task}")
        print("=" * 60)

        while not graph.is_finished(finished):

            ready = graph.ready_tasks(finished)

            if not ready:
                raise RuntimeError("Graph 存在循环依赖")

            print("\n当前可并行执行任务：")
            for node in ready:
                print(f"  - {node.name}")

            with ThreadPoolExecutor(
                max_workers=self.max_workers
            ) as executor:

                futures = {}

                for node in ready:

                    if node.name == "summary":
                        task = self._build_summary_task(
                            user_task,
                            results
                        )
                        agent = self.subagents["summary"]
                    else:
                        task = node.task
                        agent = self.subagents[node.name]

                    future = executor.submit(
                        agent.run,
                        task
                    )

                    futures[future] = node

                for future in as_completed(futures):

                    node = futures[future]

                    try:
                        result = future.result()

                        node.result = result
                        results[node.name] = result

                        finished.add(node.name)

                        print(
                            f"✓ {node.name} 执行完成"
                        )

                    except Exception as e:
                        print(
                            f"✗ {node.name} 执行失败：{e}"
                        )
                        raise

        print("\n所有 SubAgent 执行完成")

        return results

    def _build_summary_task(self, user_task, results):

        context = "\n\n".join(
            f"【{name}】\n{data['result']}"
            for name, data in results.items()
        )

        return f"""
原始任务：
{user_task}

其他SubAgent已经完成以下分析：

{context}

请综合以上结果：
1. 提炼核心结论
2. 解决原始问题
3. 给出最终方案
"""

evaluator.py:评价agent执行结果
class Evaluator:
    def evaluate(self, results):

        total = len(results)

        if total == 0:
            return {
                "score": 0,
                "comment": "没有产生任何结果"
            }

        success = 0

        for name, result in results.items():

            if (
                result.get("result")
                and len(result["result"]) > 10
            ):
                success += 1

        score = success / total * 100

        return {
            "score": round(score, 2),
            "completed": success,
            "total": total,
            "comment": self._comment(score)
        }

    def _comment(self, score):

        if score >= 90:
            return "Agent执行效果优秀"
        elif score >= 70:
            return "Agent执行效果良好"
        elif score >= 50:
            return "Agent执行效果一般"
        else:
            return "Agent执行效果较差"
