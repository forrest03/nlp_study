from openai import OpenAI
# ── JSON Schema 定义 ──────────────────────────────────────────────────
INTENT_SCHEMA = {
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
client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")

resp = client.chat.completions.create(
    model="qwen2.5-0.5b",
    messages=[{"role": "user", "content": "查茅台股价"}],
    extra_body={"guided_json": INTENT_SCHEMA},   # vLLM 扩展字段
    temperature=0,
)
print(resp.choices[0].message.content)