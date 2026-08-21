class SkillLoader:

    def __init__(self, skill_path="skills"):
        self.skill_path = skill_path
        self.skills = {}


    # 渐进式加载
    def load_skill(self, skill_name):

        if skill_name in self.skills:
            return self.skills[skill_name]


        module_name = f"{self.skill_path}.{skill_name}"

        try:
            module = importlib.import_module(module_name)

            skill = module.Skill()

            self.skills[skill_name] = skill

            print(f"[Loader] load skill: {skill_name}")

            return skill

        except Exception as e:
            print("skill加载失败:", e)
            return None

    # 根据任务寻找skill
    def search_skill(self, task):
        if "天气" in task:
            return "weather"
        elif "计算" in task or "+" in task:
            return "calculator"
        return None


class AgentHarness:
    def __init__(self):
      self.loader = SkillLoader()
    def run(self, task):
      print("Task:", task)
        # 第一步：
        # 根据任务判断需要哪个skill
        skill_name = self.loader.search_skill(task)
        if skill_name is None:
            return "没有找到对应skill"
        # 第二步：
        # 只加载需要的skill
        skill = self.loader.load_skill(skill_name)
        if skill:
            # 第三步：
            # 执行skill
            result = skill.execute(task)
            return result
        return "skill执行失败"



class Skill:
    def __init__(self):

        self.name = "weather"

    def execute(self, task):

        return "今天北京天气晴，温度25℃"



class Skill:


    def __init__(self):

        self.name="calculator"
    def execute(self, task):

        try:
            expression = task.replace("计算","")
            result = eval(expression)
            return f"计算结果:{result}"
        except:
            return "计算失败"
