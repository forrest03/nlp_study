import argparse
import json
import os
import time
from pathlib import Path
from typing import List, Dict, Optional
from sklearn.metrics import f1_score, precision_score, recall_score
import random

# 使用OpenAI客户端方式调用阿里云百炼API
from openai import OpenAI

print("1. Starting Online LLM API SFT Training (Zero-shot/One-shot/Few-shot)...")

def parse_args():
    parser = argparse.ArgumentParser(description='Online LLM API SFT Training')
    
    # 数据参数
    parser.add_argument('--data_dir', type=str, default=str(Path(__file__).parent.parent / "data" / "peoples_daily"))
    parser.add_argument('--output_dir', type=str, default=str(Path(__file__).parent.parent / "outputs" / "llm" / "online"))
    
    # 提示方式
    parser.add_argument('--prompt_type', type=str, default='few-shot', 
                        choices=['zero-shot', 'one-shot', 'few-shot'],
                        help='Prompt type: zero-shot, one-shot, or few-shot')
    parser.add_argument('--num_examples', type=int, default=3, 
                        help='Number of examples for few-shot learning')
    
    # API参数 - 默认使用阿里云百炼API
    parser.add_argument('--api_key', type=str, default=None,
                        help='API key (can also be set via DASHSCOPE_API_KEY env var)')
    parser.add_argument('--base_url', type=str, default='https://dashscope.aliyuncs.com/compatible-mode/v1',
                        help='API base URL')
    parser.add_argument('--model', type=str, default='qwen3.7-plus',
                        help='Model name for API')
    parser.add_argument('--enable_thinking', action='store_true', default=False,
                        help='Enable thinking mode')
    
    # 训练参数
    parser.add_argument('--max_retries', type=int, default=3,
                        help='Max retries for API calls')
    parser.add_argument('--retry_delay', type=int, default=2,
                        help='Delay between retries in seconds')
    
    # 标注类型
    parser.add_argument('--label_scheme', type=str, default='BIOES',
                        choices=['BIO', 'BIOES'],
                        help='Labeling scheme: BIO or BIOES')
    
    return parser.parse_args()

class OnlineLLMNER:
    """在线大模型API进行NER标注（使用阿里云百炼API）"""
    
    def __init__(self, args):
        self.args = args
        
        # 初始化OpenAI客户端（兼容阿里云百炼API）
        self.api_key = args.api_key or os.environ.get('DASHSCOPE_API_KEY')
        if not self.api_key:
            raise ValueError("API key must be provided via --api_key or DASHSCOPE_API_KEY env var")
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=args.base_url
        )
        
        # 加载数据集
        self.train_data = self._load_data(os.path.join(args.data_dir, 'train.json'))
        self.valid_data = self._load_data(os.path.join(args.data_dir, 'validation.json'))
        self.test_data = self._load_data(os.path.join(args.data_dir, 'test.json'))
        
        # 构建标签映射
        self.label2id, self.id2label = self._build_label_mapping()
        
        # 构建提示模板
        self.system_prompt = self._build_system_prompt()
        
    def _load_data(self, path: str) -> List[Dict]:
        """加载数据"""
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _build_label_mapping(self) -> tuple:
        """构建标签映射"""
        labels = set(['O'])
        for item in self.train_data:
            for tag in item['ner_tags']:
                labels.add(tag)
        label2id = {label: idx for idx, label in enumerate(sorted(labels))}
        id2label = {v: k for k, v in label2id.items()}
        return label2id, id2label
    
    def _build_system_prompt(self) -> str:
        """构建系统提示"""
        prompt = """你是一个专业的中文命名实体识别（NER）系统。你的任务是对给定的中文文本进行序列标注，识别出其中的命名实体。

标注格式：
- 对于每个字/词，标注其对应的实体类型
- 使用BIOES标注方案：
  - B-XXX: 实体的开始
  - I-XXX: 实体的中间
  - E-XXX: 实体的结束
  - S-XXX: 单字实体
  - O: 非实体

实体类型包括：PER（人名）、LOC（地名）、ORG（机构名）等

输出格式：JSON数组，每个元素对应一个字的标注，例如：
输入：["北", "京", "是", "中", "国", "的", "首", "都"]
输出：["B-LOC", "I-LOC", "O", "B-LOC", "I-LOC", "O", "O", "O"]

请只输出JSON数组，不要输出其他任何内容。
"""
        if self.args.label_scheme == 'BIO':
            prompt = prompt.replace('BIOES', 'BIO').replace('E-XXX: 实体的结束\n  - S-XXX: 单字实体\n  ', '')
        return prompt
    
    def _build_prompt(self, text: str, examples: Optional[List[Dict]] = None) -> str:
        """构建提示"""
        prompt = f"请对以下文本进行命名实体识别标注：\n\n文本：{''.join(text)}\n\n"
        
        if examples:
            prompt += "参考示例：\n"
            for i, ex in enumerate(examples, 1):
                ex_text = ''.join(ex['tokens'])
                ex_labels = ex['ner_tags']
                prompt += f"\n示例{i}：\n"
                prompt += f"文本：{ex_text}\n"
                prompt += f"标注：{json.dumps(ex_labels, ensure_ascii=False)}\n"
            prompt += "\n"
        
        prompt += "请输出标注结果（JSON数组格式）："
        return prompt
    
    def _call_api(self, messages: List[Dict]) -> str:
        """调用API（使用OpenAI客户端方式）"""
        for attempt in range(self.args.max_retries):
            try:
                # 构建额外参数
                extra_body = {}
                if self.args.enable_thinking:
                    extra_body["enable_thinking"] = True
                
                # 调用API
                completion = self.client.chat.completions.create(
                    model=self.args.model,
                    messages=messages,
                    extra_body=extra_body,
                    stream=False  # 非流式调用
                )
                
                return completion.choices[0].message.content
            
            except Exception as e:
                print(f"API call failed (attempt {attempt + 1}/{self.args.max_retries}): {str(e)}")
                if attempt < self.args.max_retries - 1:
                    time.sleep(self.args.retry_delay)
                    continue
                raise e
    
    def _get_examples(self, data: List[Dict], num: int) -> List[Dict]:
        """获取示例数据"""
        return random.sample(data, min(num, len(data)))
    
    def predict(self, tokens: List[str], examples: Optional[List[Dict]] = None) -> List[str]:
        """预测标注"""
        # 构建提示
        if self.args.prompt_type == 'zero-shot':
            examples = None
        elif self.args.prompt_type == 'one-shot':
            examples = self._get_examples(self.train_data, 1)
        elif self.args.prompt_type == 'few-shot':
            examples = self._get_examples(self.train_data, self.args.num_examples)
        
        prompt = self._build_prompt(tokens, examples)
        
        # 调用API
        messages = [
            {'role': 'system', 'content': self.system_prompt},
            {'role': 'user', 'content': prompt}
        ]
        
        response = self._call_api(messages)
        
        # 解析响应
        try:
            labels = json.loads(response)
            if not isinstance(labels, list):
                raise ValueError("Response is not a list")
            return labels
        except:
            # 如果解析失败，返回全O
            print(f"Failed to parse response: {response}")
            return ['O'] * len(tokens)
    
    def evaluate(self, data: List[Dict], split_name: str = 'test') -> Dict:
        """评估数据集"""
        print(f"\n{'='*50}")
        print(f"Evaluating on {split_name} set ({self.args.prompt_type})")
        print(f"{'='*50}")
        
        all_preds = []
        all_labels = []
        decoded_samples = []
        
        # 准备示例
        if self.args.prompt_type == 'one-shot':
            examples = self._get_examples(self.train_data, 1)
        elif self.args.prompt_type == 'few-shot':
            examples = self._get_examples(self.train_data, self.args.num_examples)
        else:
            examples = None
        
        for i, item in enumerate(data):
            tokens = item['tokens']
            true_labels = item['ner_tags']
            
            # 预测
            if examples is None:
                pred_labels = self.predict(tokens)
            else:
                pred_labels = self.predict(tokens, examples)
            
            # 对齐长度
            pred_labels = pred_labels[:len(tokens)]
            if len(pred_labels) < len(tokens):
                pred_labels += ['O'] * (len(tokens) - len(pred_labels))
            
            # 转换为ID
            pred_ids = [self.label2id.get(l, self.label2id['O']) for l in pred_labels]
            true_ids = [self.label2id.get(l, self.label2id['O']) for l in true_labels]
            
            all_preds.extend(pred_ids)
            all_labels.extend(true_ids)
            
            # 解码实体
            entities = self._decode_entities(tokens, pred_labels)
            decoded_samples.append({
                'tokens': tokens,
                'predicted_entities': entities,
                'original_tags': true_labels,
                'predicted_tags': pred_labels
            })
            
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(data)} samples")
        
        # 计算指标
        if len(all_labels) > 0:
            f1 = f1_score(all_labels, all_preds, average='macro')
            precision = precision_score(all_labels, all_preds, average='macro')
            recall = recall_score(all_labels, all_preds, average='macro')
        else:
            f1 = precision = recall = 0.0
        
        results = {
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'decoded_samples': decoded_samples
        }
        
        print(f"\n{split_name} Results ({self.args.prompt_type}):")
        print(f"F1: {f1:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        
        return results
    
    def _decode_entities(self, tokens: List[str], labels: List[str]) -> List[Dict]:
        """解码实体"""
        entities = []
        current_entity = None
        
        for i, (token, label) in enumerate(zip(tokens, labels)):
            if label.startswith('B-'):
                if current_entity:
                    entities.append(current_entity)
                current_entity = {
                    'text': token,
                    'label': label[2:],
                    'start': i,
                    'end': i + 1
                }
            elif label.startswith('I-') and current_entity:
                current_entity['text'] += token
                current_entity['end'] = i + 1
            elif label.startswith('E-') and current_entity:
                current_entity['text'] += token
                current_entity['end'] = i + 1
                entities.append(current_entity)
                current_entity = None
            elif label.startswith('S-'):
                entities.append({
                    'text': token,
                    'label': label[2:],
                    'start': i,
                    'end': i + 1
                })
            elif label == 'O':
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
        
        if current_entity:
            entities.append(current_entity)
        
        return entities

def main():
    args = parse_args()
    
    print("\n" + "=" * 60)
    print("Online LLM API SFT Training Configuration")
    print("=" * 60)
    for arg, value in vars(args).items():
        if arg == 'api_key':
            print(f"{arg}: {'***' if value else 'None'}")
        else:
            print(f"{arg}: {value}")
    print("=" * 60)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 初始化NER系统
    print("\nInitializing Online LLM NER System...")
    ner = OnlineLLMNER(args)
    
    print(f"\nDataset sizes:")
    print(f"Training: {len(ner.train_data)}")
    print(f"Validation: {len(ner.valid_data)}")
    print(f"Test: {len(ner.test_data)}")
    print(f"Labels: {ner.label2id}")
    
    # 评估
    results = {}
    
    # 验证集
    valid_results = ner.evaluate(ner.valid_data, 'validation')
    results['validation'] = valid_results
    
    # 测试集
    test_results = ner.evaluate(ner.test_data, 'test')
    results['test'] = test_results
    
    # 保存结果
    output_data = {
        'configuration': vars(args),
        'label_mapping': {
            'label2id': ner.label2id,
            'id2label': ner.id2label
        },
        'results': {
            'validation': {
                'f1': valid_results['f1'],
                'precision': valid_results['precision'],
                'recall': valid_results['recall']
            },
            'test': {
                'f1': test_results['f1'],
                'precision': test_results['precision'],
                'recall': test_results['recall']
            }
        },
        'validation_decoded_samples': valid_results['decoded_samples'],
        'test_decoded_samples': test_results['decoded_samples']
    }
    
    output_path = os.path.join(args.output_dir, f'sft_results_{args.prompt_type}.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n" + "=" * 60)
    print("SFT Training Summary (Online API - 阿里云百炼)")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Prompt Type: {args.prompt_type}")
    print(f"Validation F1: {valid_results['f1']:.4f}")
    print(f"Test F1: {test_results['f1']:.4f}")
    print(f"\nResults saved to: {output_path}")
    print("=" * 60)

if __name__ == '__main__':
    main()