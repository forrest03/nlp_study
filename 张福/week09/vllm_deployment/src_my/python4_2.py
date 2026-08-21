import json
import inspect
from vllm import LLM, SamplingParams
try:
    from vllm.sampling_params import GuidedDecodingParams
    has_guided_decoding = True
except ImportError:
    has_guided_decoding = False

llm = LLM(model="/root/autodl-tmp/Qwen2___5-0___5B-Instruct",
          max_model_len=2048, gpu_memory_utilization=0.6)

schema = {
    "type": "object",
    "properties": {
        "company": {
            "type": "string",
            "description": "公司全称，如 招商银行、贵州茅台",
        },
        "year": {
            "type": "integer",
            "minimum": 2015,
            "maximum": 2025,
        },
        "metric": {
            "type": "string",
            "enum": ["营收", "净利润", "ROE", "毛利率", "总资产", "经营现金流"],
        },
    },
    "required": ["company", "year", "metric"],
    "additionalProperties": False,
}

guided_params = None
if has_guided_decoding:
    sig = inspect.signature(GuidedDecodingParams.__init__)
    params = list(sig.parameters.keys())
    if "json_schema" in params:
        guided_params = GuidedDecodingParams(json_schema=schema)
    elif "json" in params:
        guided_params = GuidedDecodingParams(json=schema)
    else:
        print(f"GuidedDecodingParams 参数: {params}")
        has_guided_decoding = False

system_prompt = f"""你是一个金融数据提取助手。请根据用户查询，提取公司名称、年份和财务指标，严格按照以下JSON schema格式输出，不要包含任何其他文字：
{json.dumps(schema, ensure_ascii=False)}"""

messages_list = [
    [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "查茅台股价"}
    ],
    [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "贵州茅台2024年营收"}
    ]
]

sampling_params = SamplingParams(
    temperature=0,
    max_tokens=256,
    guided_decoding=guided_params
)

outputs = llm.chat(messages_list, sampling_params)
for i, output in enumerate(outputs):
    text = output.outputs[0].text
    print(f"\n=== 输出 {i+1} ===")
    print(f"原始文本: {text}")
    try:
        data = json.loads(text)
        print(f"解析后数据: {json.dumps(data, ensure_ascii=False, indent=2)}")
        print(f"字段校验:")
        for field in schema.get("required", []):
            if field in data:
                print(f"  ✓ {field}: {data[field]}")
            else:
                print(f"  ✗ {field}: 缺失")
    except json.JSONDecodeError as e:
        print(f"JSON解析失败: {e}")
        print("尝试提取文本中的JSON部分...")
        start_idx = text.find("{")
        end_idx = text.rfind("}") + 1
        if start_idx != -1 and end_idx != -1:
            json_str = text[start_idx:end_idx]
            try:
                data = json.loads(json_str)
                print(f"提取并解析成功: {json.dumps(data, ensure_ascii=False, indent=2)}")
            except json.JSONDecodeError:
                print("提取的JSON部分仍无法解析")