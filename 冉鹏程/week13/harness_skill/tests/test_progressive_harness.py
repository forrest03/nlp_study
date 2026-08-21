"""覆盖渐进式阶段、输入边界和中文路由的单元测试。"""

from pathlib import Path
import unittest

from progressive_harness import InvalidReferenceError, ProgressiveHarness

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SKILLS_ROOT = PROJECT_ROOT / "skills"


class ProgressiveHarnessTests(unittest.TestCase):
    """验证阶段顺序和可信本地文件边界。"""

    def setUp(self) -> None:
        """基于仓库中复制的示例 skill 创建一个全新的 Harness。

        参数：
            无。

        返回：
            无。
        """
        self.harness = ProgressiveHarness(SKILLS_ROOT)

    def test_discovery_reads_diagram_metadata(self) -> None:
        """第一阶段暴露名称和说明，但不加载完整说明。

        参数：
            无。

        返回：
            无。
        """
        metadata = self.harness.discover()
        self.assertEqual([item.name for item in metadata], ["baoyu-diagram"])
        self.assertIn("SVG", metadata[0].description)

    def test_chinese_request_selects_diagram_skill(self) -> None:
        """第二阶段会为中文架构图请求选中已复制的 skill。

        参数：
            无。

        返回：
            无。
        """
        metadata = self.harness.discover()
        candidates = self.harness.select("画一个订单系统架构图", metadata)
        self.assertEqual(candidates[0].metadata.name, "baoyu-diagram")
        self.assertGreater(candidates[0].score, 0)

    def test_hydration_loads_instructions_only_after_selection(self) -> None:
        """第三阶段会按需返回原始完整 Markdown 说明。

        参数：
            无。

        返回：
            无。
        """
        metadata = self.harness.discover()
        loaded_skill = self.harness.load_skill("baoyu-diagram", metadata)
        self.assertIn("# 图表生成器", loaded_skill.instructions)

    def test_explicit_reference_loads_known_architecture_guide(self) -> None:
        """第四阶段仅在显式请求时读取已列出的引用文件。

        参数：
            无。

        返回：
            无。
        """
        metadata = self.harness.discover()
        reference = self.harness.load_reference("baoyu-diagram", "architecture.md", metadata)
        self.assertIn("架构图", reference.content)

    def test_reference_path_traversal_is_rejected(self) -> None:
        """外部引用输入不能逃逸出被选 skill 的目录。

        参数：
            无。

        返回：
            无。
        """
        metadata = self.harness.discover()
        with self.assertRaises(InvalidReferenceError):
            self.harness.load_reference("baoyu-diagram", "../SKILL.md", metadata)

    def test_blank_request_is_rejected(self) -> None:
        """路由会在匹配前拒绝为空的外部请求。

        参数：
            无。

        返回：
            无。
        """
        with self.assertRaises(ValueError):
            self.harness.select("   ", self.harness.discover())
