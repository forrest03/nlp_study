'''
    当前程序为智能体的模型配置页面，当用户输入"/model"时，进入当前的配置交互页面
    # 输入baseUrl 作为模型的地址,输入apiKey 作为模型的密钥 
    # 回车后选择选择哪个模型（api获取所有模型）
    # 输入模型的编号，回车后选择该模型
    # 确认选择模型后全局加载当前模型
    # 模型存储在本地的配置文件中
'''

import json
import os
from pathlib import Path
from typing import Optional, Dict, List
from openai import OpenAI


# 配置文件路径（当前项目目录下）
CONFIG_DIR = Path(__file__).parent
CONFIG_FILE = CONFIG_DIR / ".agent_config"


class ModelConfig:
    """模型配置管理类"""
    
    def __init__(self):
        self.base_url: Optional[str] = None
        self.api_key: Optional[str] = None
        self.model_name: Optional[str] = None
        self.client: Optional[OpenAI] = None
        
    def save_config(self):
        """保存配置到本地文件"""
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        
        config = {
            "base_url": self.base_url,
            "api_key": self.api_key,
            "model_name": self.model_name
        }
        
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        print(f"✓ 配置已保存到: {CONFIG_FILE}")
    
    def load_config(self) -> bool:
        """从本地文件加载配置"""
        if not CONFIG_FILE.exists():
            return False
        
        try:
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            self.base_url = config.get("base_url")
            self.api_key = config.get("api_key")
            self.model_name = config.get("model_name")
            
            # 如果配置完整，初始化客户端
            if self.base_url and self.api_key and self.model_name:
                self._init_client()
                return True
            
            return False
        except Exception as e:
            print(f"✗ 加载配置失败: {e}")
            return False
    
    def _init_client(self):
        """初始化 OpenAI 客户端"""
        if self.base_url and self.api_key:
            self.client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key
            )
    
    def get_models(self) -> List[str]:
        """从 API 获取可用模型列表"""
        if not self.client:
            raise ValueError("客户端未初始化")
        
        try:
            models = self.client.models.list()
            return [model.id for model in models.data]
        except Exception as e:
            raise Exception(f"获取模型列表失败: {e}")
    
    def is_configured(self) -> bool:
        """检查是否已配置完整"""
        return bool(self.base_url and self.api_key and self.model_name and self.client)


# 全局配置实例
_global_config = ModelConfig()


def get_global_config() -> ModelConfig:
    """获取全局配置实例"""
    return _global_config


def interactive_config():
    """交互式配置模型"""
    print("\n" + "="*60)
    print("模型配置向导")
    print("="*60)
    
    # 1. 输入 Base URL
    print("\n请输入模型 API 地址 (Base URL):")
    print("  示例: https://api.openai.com/v1")
    print("  示例: https://api.deepseek.com/v1")
    base_url = input("> ").strip()
    
    if not base_url:
        print("✗ Base URL 不能为空")
        return False
    
    # 2. 输入 API Key
    print("\n请输入 API Key:")
    api_key = input("> ").strip()
    
    if not api_key:
        print("✗ API Key 不能为空")
        return False
    
    # 3. 临时初始化客户端，获取模型列表
    print("\n正在获取可用模型列表...")
    temp_client = OpenAI(base_url=base_url, api_key=api_key)
    
    try:
        models = temp_client.models.list()
        model_ids = [model.id for model in models.data]
        
        if not model_ids:
            print("✗ 未找到可用模型")
            return False
        
        # 4. 显示模型列表
        print(f"\n找到 {len(model_ids)} 个可用模型:")
        print("-" * 60)
        for idx, model_id in enumerate(model_ids, 1):
            print(f"  [{idx}] {model_id}")
        print("-" * 60)
        
        # 5. 选择模型
        print("\n请输入模型编号 (直接回车使用第一个):")
        choice = input("> ").strip()
        
        if not choice:
            model_idx = 0
        else:
            try:
                model_idx = int(choice) - 1
                if model_idx < 0 or model_idx >= len(model_ids):
                    print(f"✗ 编号超出范围 (1-{len(model_ids)})")
                    return False
            except ValueError:
                print("✗ 请输入有效的数字")
                return False
        
        selected_model = model_ids[model_idx]
        
        # 6. 确认选择
        print(f"\n已选择模型: {selected_model}")
        print("确认保存配置? (y/n):")
        confirm = input("> ").strip().lower()
        
        if confirm != 'y':
            print("✗ 配置已取消")
            return False
        
        # 7. 保存到全局配置
        _global_config.base_url = base_url
        _global_config.api_key = api_key
        _global_config.model_name = selected_model
        _global_config._init_client()
        _global_config.save_config()
        
        print("\n✓ 模型配置完成!")
        print(f"  Base URL: {base_url}")
        print(f"  Model: {selected_model}")
        
        return True
        
    except Exception as e:
        print(f"✗ 获取模型列表失败: {e}")
        return False


def show_current_config():
    """显示当前配置"""
    if _global_config.is_configured():
        print("\n当前模型配置:")
        print(f"  Base URL: {_global_config.base_url}")
        print(f"  Model: {_global_config.model_name}")
        print(f"  API Key: {_global_config.api_key[:10]}...")
    else:
        print("\n未配置模型，请输入 /model 进行配置")


# 程序启动时自动加载配置
_global_config.load_config()


if __name__ == "__main__":
    # 测试：运行交互式配置
    interactive_config()
    show_current_config()
