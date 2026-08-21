### 2.2 验证可用

新开一个终端：

```bash
# 查询已加载模型
curl http://localhost:8000/v1/models

# 简单对话
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen2.5-0.5b",
    "messages": [{"role": "user", "content": "你好"}],
    "max_tokens": 50
  }'

#回复
curl http://localhost:8000/v1/chat/completions \
>   -H 'Content-Type: application/json' \
>   -d '{
>     "model": "qwen2.5-0.5b",
>     "messages": [{"role": "user", "content": "你好"}],
>     "max_tokens": 50
>   }'
{"id":"chatcmpl-cc49614c6b34469b8c15d81a5d65d04d","object":"chat.completion","created":1783660592,"model":"qwen2.5-0.5b","choices":[{"index":0,"message":{"role":"assistant","reasoning_content":null,"content":"你好！很高兴为你服务。如果你有任何问题或需要帮助，请随时告诉我，我会尽力提供支持和解答。","tool_calls":[]},"logprobs":null,"finish_reason":"stop","stop_reason":null}],"usage":{"prompt_tokens":30,"total_tokens":54,"completion_tokens":24,"prompt_tokens_details":null},"prompt_logprobs":null,"kv_transfer_params":null}
```

---
## 三、各脚本使用方法

所有脚本位于 `src/`，运行前确保已激活 venv 且 server 已启动（`bench_throughput.py` 除外）。

### 3.1 demo_guided_choice.py — 枚举约束

```bash
cd src/
python demo_guided_choice.py
```

**场景**：金融问答意图路由（查股价 / 查财报 / 查新闻 / 对比分析 / 其他）

**内部流程**：
1. 对每个测试问题，调用 server 两次：裸 prompt + `extra_body={"guided_choice": ...}`
2. 对比两种模式的输出合法率和分类准确率

**预期输出**（关键行）：
```
输出合法（在枚举内）   10/12 (83%)    12/12 (100%)
预测正确            3/12 (25%)     3/12 (25%)
```
**实际输出结果**:
```
python demo_guided_choice.py 
======================================================================
  Demo: guided_choice（枚举约束）
  Model: qwen2.5-0.5b   Choices: ['查股价', '查财报', '查新闻', '对比分析', '其他']
======================================================================

问题                              真值        裸 prompt 输出         guided 输出      
------------------------------------------------------------------------------------------
查一下茅台今天多少钱                      查股价       ✓ 查股价               ✓ 查股价
贵州茅台 2024 年营收多少亿                查财报       ~ 其他                ✗ 其他
最近宁德时代有什么新闻                     查新闻       ~ 其他                ✗ 其他
对比一下招行和平安的净利润                   对比分析      ~ 其他                ✗ 其他
今天天气怎么样                         其他        ✓ 其他                ✓ 其他
帮我看看 600000 的收盘价                查股价       ✓ 查股价               ✓ 查股价
招商银行去年的 ROE 是多少                 查财报       ~ 查股价               ✗ 查股价
宁德时代被限产了吗                       查新闻       ~ 对比分析              ✗ 对比分析
比亚迪和特斯拉哪个更强                     对比分析      ✓ 对比分析              ✓ 对比分析
帮我订一张机票                         其他        ~ 查股价               ✗ 查股价
五粮液现在股价                         查股价       ✓ 查股价               ✓ 查股价
平安保险的净利润增长率                     查财报       ~ 对比分析              ✗ 对比分析
------------------------------------------------------------------------------------------

指标                            裸 prompt            guided_choice       
----------------------------------------------------------------------
输出合法（在枚举内）              12/12 (100%)        12/12 (100%)
预测正确                        5/12 (42%)           5/12 (42%)
平均延迟（秒）                     0.047             0.197

======================================================================
  结论：guided_choice 100% 保证输出合法，
       分类准确率也通常比裸 prompt 高（因为模型不会被错误 token 带偏）
======================================================================
```
---
### 3.2 demo_guided_regex.py — 正则约束

```bash
python demo_guided_regex.py
```

**场景**：日期标准化（→ YYYY-MM-DD）、股票代码抽取（→ 6 位数字）

**教学要点**：凡下游有严格解析器的字段，正则约束能把"模型说对但格式错"的问题一次根治。
**实际输出结果**
```
python demo_guided_regex.py 
======================================================================
  任务 1：日期标准化 → YYYY-MM-DD
  正则: \d{4}-\d{2}-\d{2}
======================================================================

输入                                 裸 prompt                 guided_regex   
---------------------------------------------------------------------------
2024年5月12日                         ✓ 2024-05-12            ✓ 2024-05-12
2023/12/1 下午开会                     ✗ 2023-12-01 15:00:00   ✓ 2023-12-01
三月三号我去北京                           ✗ 03-03                 ✓ 0331-00-00
2024.11.30 是截止日期                   ✓ 2024-11-30            ✓ 2024-11-30
明天（假设今天是2026-05-11）                ✓ 2026-05-13            ✓ 2026-05-13
2024 年 10 月的第一天                    ✗ 01-01-2024            ✓ 0109-01-01
---------------------------------------------------------------------------
格式合法率：裸 prompt 3/6 (50%)  |  guided_regex 6/6 (100%)

======================================================================
  任务 2：A 股代码抽取 → 6 位数字
  正则: \d{6}
======================================================================

输入                                 裸 prompt                 guided_regex   
---------------------------------------------------------------------------
帮我查 600000 浦发银行                    ✓ 600300                ✓ 600300
code: 000001 平安银行                  ✓ 000001                ✓ 000001
茅台的代码是 600519                      ✓ 600519                ✓ 600519
六零零五一九                             ✓ 600595                ✓ 600595
股票代码：300750（宁德时代）                  ✓ 300750                ✓ 300750
---------------------------------------------------------------------------
格式合法率：裸 prompt 5/5 (100%)  |  guided_regex 5/5 (100%)

======================================================================
  结论：guided_regex 保证下游解析器永远能拿到合法输入
       特别适合日期/电话/代码/邮编等有严格格式的字段
======================================================================
```
---
### 3.3 demo_guided_json.py — JSON Schema 基础

```bash
python demo_guided_json.py
```

**场景**：财报问答意图抽取（公司/年度/指标三元组）

**三种模式对比**：
- 裸 prompt：靠指令和 few-shot
- `response_format={"type": "json_object"}`：OpenAI 标准，保证是 JSON
- `guided_json=schema`：vLLM 扩展，保证完全符合 Schema

**关键看点**："22 年" 这类输入下，裸 prompt 和 response_format 可能输出 `year: 22`（违反 `minimum: 2015`），只有 guided_json 能强制修正为 2022。
**实际运行结果**
```
python demo_guided_json.py 
==============================================================================
  Demo: guided_json（JSON Schema 约束）
  Model: qwen2.5-0.5b
  对比三种模式：裸 prompt / response_format / guided_json
==============================================================================

▶ 招行 2023 年营收多少
  [raw             ] ✓  {"company": "招行", "year": 2023, "metric": "营收"}
  [response_format ] ✓  {"company": "招行", "year": 2023, "metric": "营收"}
  [guided_json     ] ✓  {"company": "招行", "year": 2023, "metric": "营收"}

▶ 贵州茅台 2022 的净利润
  [raw             ] ✓  {"company": "贵州茅台", "year": 2022, "metric": "净利润"}
  [response_format ] ✓  {"company": "贵州茅台", "year": 2022, "metric": "净利润"}
  [guided_json     ] ✓  {"company": "贵州茅台", "year": 2022, "metric": "净利润"}

▶ 平安银行去年（2024）的 ROE
  [raw             ] ✓  {"company": "平安银行", "year": 2024, "metric": "ROE"}
  [response_format ] ✓  {"company": "平安银行", "year": 2024, "metric": "ROE"}
  [guided_json     ] ✓  {"company": "平安银行", "year": 2024, "metric": "ROE"}

▶ 2021 年五粮液毛利率
  [raw             ] ✓  {"company": "五粮液", "year": 2021, "metric": "毛利率"}
  [response_format ] ✓  {"company": "五粮液", "year": 2021, "metric": "毛利率"}
  [guided_json     ] ✓  {"company": "五粮液", "year": 2021, "metric": "毛利率"}

▶ 2023 宁德时代经营现金流
  [raw             ] ✓  {"company": "宁德时代", "year": 2023, "metric": "经营现金流"}
  [response_format ] ✓  {"company": "宁德时代", "year": 2023, "metric": "经营现金流"}
  [guided_json     ] ✓  {"company": "宁德时代", "year": 2023, "metric": "经营现金流"}

▶ 问一下比亚迪 2024 的总资产规模
  [raw             ] ✓  {"company": "比亚迪股份有限公司", "year": 2024, "metric": "总资产"}
  [response_format ] ✓  {"company": "比亚迪股份有限公司", "year": 2024, "metric": "总资产"}
  [guided_json     ] ✓  {"company": "比亚迪股份有限公司", "year": 2024, "metric": "总资产"}

▶ 茅台 2020 年利润情况
  [raw             ] ✓  {"company": "茅台酒股份有限公司", "year": 2020, "metric": "净利润"}
  [response_format ] ✓  {"company": "茅台酒股份有限公司", "year": 2020, "metric": "净利润"}
  [guided_json     ] ✓  {"company": "茅台酒股份有限公司", "year": 2020, "metric": "净利润"}

▶ ICBC 2023 营收
  [raw             ] ✗  {"company": "ICBC", "year": 2023, "metric": "营业收入"}
  [response_format ] ✗  {"company": "ICBC", "year": 2023, "metric": "营业收入"}
  [guided_json     ] ✓  {"company": "ICBC", "year": 2023, "metric": "营收"}

▶ 隆基绿能 22 年 roe
  [raw             ] ✓  {"company": "隆基绿能", "year": 2022, "metric": "ROE"}
  [response_format ] ✓  {"company": "隆基绿能", "year": 2022, "metric": "ROE"}
  [guided_json     ] ✓  {"company": "隆基绿能", "year": 2022, "metric": "ROE"}

==============================================================================
  9 条测试结果汇总
==============================================================================
指标                      裸 prompt          response_format     guided_json    
------------------------------------------------------------------------------
合法 JSON               9/9 (100%)      9/9 (100%)      9/9 (100%)      
字段齐全                  9/9 (100%)      9/9 (100%)      9/9 (100%)      
year 在 2015~2025      9/9 (100%)      9/9 (100%)      9/9 (100%)      
metric 在枚举内           8/9 (89%)      8/9 (89%)      9/9 (100%)      
jsonschema 完全通过       8/9 (89%)      8/9 (89%)      9/9 (100%)      

==============================================================================
  结论：
    response_format 只保证是 JSON，不保证字段名、类型、枚举正确
    guided_json     是唯一 100% 保证 schema 合法的方式
==============================================================================
```
---
### 3.4 demo_response_format.py — OpenAI 标准方式

```bash
python demo_response_format.py
```

**场景**：新闻情感分类 + 置信度 + 关键词

**教学要点**：`response_format={"type": "json_object"}` 是 OpenAI/Azure/vLLM 都兼容的**可移植方案**。相比 `guided_json` 它跨厂商可用但约束更弱。选型时权衡：
- 跨厂商部署 → response_format
- 单一 vLLM 部署 + 严格解析 → guided_json

**实际运行结果**
```
python demo_response_format.py 
===========================================================================
  Demo: response_format（OpenAI 标准 JSON 模式）
  Model: qwen2.5-0.5b
===========================================================================

▶ 茅台三季度营收创历史新高，净利润同比增长 15%
  [raw         ] ✓ {
  "sentiment": "positive",
  "confidence": 0.9,
  "keywords": ["茅台", "三季度", "营收", "净利润", "增长"]
}
  [json_object ] ✓ {
  "sentiment": "positive",
  "confidence": 0.9,
  "keywords": ["茅台", "三季度", "营收", "净利润", "增长"]
}

▶ 比亚迪召回 10 万辆电动车，涉及电池安全问题
  [raw         ] ✓ {
  "sentiment": "negative",
  "confidence": 0.85,
  "keywords": ["召回", "电池安全"]
}
  [json_object ] ✓ {
  "sentiment": "negative",
  "confidence": 0.85,
  "keywords": ["召回", "电池安全"]
}

▶ 央行维持 LPR 利率不变
  [raw         ] ✓ {
  "sentiment": "neutral",
  "confidence": 0.75,
  "keywords": ["LPR", "保持不变"]
}
  [json_object ] ✓ {
  "sentiment": "neutral",
  "confidence": 0.75,
  "keywords": ["LPR", "保持不变"]
}

▶ 宁德时代与宝马签订长期供货协议
  [raw         ] ✓ {
  "sentiment": "positive",
  "confidence": 0.95,
  "keywords": ["宁德时代", "宝马", "长期供货协议"]
}
  [json_object ] ✓ {
  "sentiment": "positive",
  "confidence": 0.95,
  "keywords": ["宁德时代", "宝马", "长期供货协议"]
}

▶ 平安保险高管被调查，股价下跌 8%
  [raw         ] ✓ {"sentiment": "negative", "confidence": 0.9, "keywords": ["平安保险"]}
  [json_object ] ✓ {"sentiment": "negative", "confidence": 0.9, "keywords": ["平安保险"]}

===========================================================================
  5 条测试结果
===========================================================================
指标                    裸 prompt            response_format     
------------------------------------------------------------
合法 JSON             5/5 (100%)      5/5 (100%)      
有 sentiment 字段      5/5 (100%)      5/5 (100%)      
sentiment 值合法       5/5 (100%)      5/5 (100%)      
有 confidence 字段     5/5 (100%)      5/5 (100%)      
有 keywords 字段       5/5 (100%)      5/5 (100%)      

===========================================================================
  观察：
    response_format 显著提升 JSON 合法率，但字段语义仍靠模型自觉
    若需严格字段 schema，请用 guided_json（见 demo_function_call.py）
===========================================================================
```
---
### 3.5 demo_function_call.py ★ 核心

```bash
# 跑两个工具共 100 个用例
python demo_function_call.py

# 只跑一个
python demo_function_call.py --tool stock
python demo_function_call.py --tool order
```

**两个工具**：
- `get_stock_quote`：金融股价查询，schema 含 string+enum+regex+array+minItems
- `create_order`：电商下单，schema 含 integer 范围+手机号正则+多枚举

**每个工具 50 条测试**，三种模式对比，产出：
- 终端表格：JSON 合法率 / 必选字段率 / schema 完全通过率
- 典型失败案例（前 3 条）
- `outputs/function_call_results.json`：详细数据（可用于后续分析）

**预期结果**：
| 指标 | 裸 prompt | response_format | guided_json |
|------|----------|-----------------|-------------|
| JSON 合法 | ~90% | 100% | 100% |
| 字段齐全 | ~90% | 100% | 100% |
| **完整 schema 通过** | **40-60%** | **40-70%** | **100%** |

**核心教学点**：`response_format` 和 `guided_json` 之间的 30~50 个百分点差距就是约束解码的工程价值——`response_format` 只管语法，不管字段值是否合法。
**实际运行结果**
```
python demo_function_call.py --tool stock
==============================================================================
  demo_function_call.py   核心：裸 prompt vs response_format vs guided_json
  Model: qwen2.5-0.5b
==============================================================================

==============================================================================
  工具: get_stock_quote   测试数: 50   模式: 3
==============================================================================
  进度: 10/50
  进度: 20/50
  进度: 30/50
  进度: 40/50
  进度: 50/50

──────────────────────────────────────────────────────────────────────────────
  【get_stock_quote】 50 条测试 × 3 模式 汇总
──────────────────────────────────────────────────────────────────────────────
指标                      裸 prompt            response_format       guided_json    
──────────────────────────────────────────────────────────────────────────────
JSON 语法合法             50/50 (100%)       50/50 (100%)       50/50 (100%)       
必选字段齐全                50/50 (100%)       50/50 (100%)       50/50 (100%)       
完整 schema 通过 ★        46/50 ( 92%)       46/50 ( 92%)       50/50 (100%)       
平均延迟（秒）               0.296              0.305              0.307              

──────────────────────────────────────────────────────────────────────────────
  【get_stock_quote】 典型失败案例（前 3 条）
──────────────────────────────────────────────────────────────────────────────

[raw] 失败示例（schema 校验未通过）：
  ▶ Prompt: 帮我查询平安银行今日开盘价，并简单解释什么是开盘价
    输出:   {"symbol": "02699", "market": "SH", "date": "2023-07-10", "fields": ["open"], "adjust": "none"}
    错误:   schema: '02699' does not match '^\\d{6}$'
  ▶ Prompt: 请查询东方财富成交量并分析异动原因
    输出:   {"symbol": "000001.SZ", "market": "SZ", "date": "2023-10-07", "fields": ["volume"], "adjust": "hfq", "symbols": ["000001.SZ"]}
    错误:   schema: Additional properties are not allowed ('symbols' was unexpected)
  ▶ Prompt: 帮我查 000001 收盘价，然后帮我判断是否该买入
    输出:   {"symbol": "000001", "market": "SH", "date": "2023-07-15", "fields": ["close"], "adjust": "none", "isBuy": true}
    错误:   schema: Additional properties are not allowed ('isBuy' was unexpected)

[response_format] 失败示例（schema 校验未通过）：
  ▶ Prompt: 帮我查询平安银行今日开盘价，并简单解释什么是开盘价
    输出:   {"symbol": "02699", "market": "SH", "date": "2023-07-10", "fields": ["open"], "adjust": "none"}
    错误:   schema: '02699' does not match '^\\d{6}$'
  ▶ Prompt: 请查询东方财富成交量并分析异动原因
    输出:   {"symbol": "000001.SZ", "market": "SZ", "date": "2023-10-07", "fields": ["volume"], "adjust": "hfq", "symbols": ["000001.SZ"]}
    错误:   schema: Additional properties are not allowed ('symbols' was unexpected)
  ▶ Prompt: 帮我查 000001 收盘价，然后帮我判断是否该买入
    输出:   {"symbol": "000001", "market": "SH", "date": "2023-07-15", "fields": ["close"], "adjust": "none", "isBuy": true}
    错误:   schema: Additional properties are not allowed ('isBuy' was unexpected)

[guided_json] ✓ 无失败案例

  [耗时 46.5s]

详细结果已保存：/root/autodl-tmp/vllm_deployment/src/../outputs/function_call_results.json

==============================================================================
  核心结论：
    裸 prompt        — JSON 语法偶尔错 / 字段拼错 / 正则枚举不符
    response_format  — JSON 合法率接近满分，但字段语义仍错
    guided_json      — 100% 满足完整 schema（小模型从不可用变可靠）
==============================================================================
```
---
### 3.6 bench_throughput.py — 吞吐对比

```bash
# 先停 vLLM server（需要释放显存）
fuser -k 8000/tcp

python bench_throughput.py
```

**三种路线**：
- [A] transformers 串行（一次一条）
- [B] transformers batch=8（手动 padding）
- [C] vLLM 批处理（内置 continuous batching）

**产出**：
- 终端表格：总耗时 / QPS / tokens/s / 相对 vLLM 的倍率
- `outputs/throughput_comparison.png`：三路对比柱状图
- `outputs/throughput_results.json`：详细数据

**预期倍率**（Qwen2-0.5B / RTX 4060 8GB）：
- 串行 ≈ 基准 1×
- batch=8 ≈ 2~4×
- vLLM ≈ 5~15×（视请求多样性）
**实际运行结果**
```
python bench_throughput_my.py 
======================================================================
  Throughput Benchmark  |  50 prompts × max 100 new tokens
======================================================================

======================================================================
  加载 transformers Qwen2.5-0.5B-Instruct
======================================================================

[A] transformers 串行（一次一条）...
    进度 10/50
    进度 20/50
    进度 30/50
    进度 40/50
    进度 50/50

[B] transformers batch=8（手动 padding）...
    进度 batch 1/7
    进度 batch 2/7
    进度 batch 3/7
    进度 batch 4/7
    进度 batch 5/7
    进度 batch 6/7
    进度 batch 7/7

======================================================================
  加载 vLLM Qwen2.5-0.5B-Instruct
======================================================================
INFO 07-10 14:26:56 [__init__.py:244] Automatically detected platform cuda.
INFO 07-10 14:27:03 [config.py:841] This model supports multiple tasks: {'generate', 'reward', 'classify', 'embed'}. Defaulting to 'generate'.
WARNING 07-10 14:27:03 [config.py:3371] Casting torch.bfloat16 to torch.float16.
INFO 07-10 14:27:03 [config.py:1472] Using max model len 2048
INFO 07-10 14:27:03 [config.py:2285] Chunked prefill is enabled with max_num_batched_tokens=8192.
WARNING 07-10 14:27:03 [cuda.py:102] To see benefits of async output processing, enable CUDA graph. Since, enforce-eager is enabled, async output processor cannot be used
WARNING 07-10 14:27:03 [__init__.py:2662] We must use the `spawn` multiprocessing start method. Overriding VLLM_WORKER_MULTIPROC_METHOD to 'spawn'. See https://docs.vllm.ai/en/latest/usage/troubleshooting.html#python-multiprocessing for more information. Reason: CUDA is initialized
INFO 07-10 14:27:07 [__init__.py:244] Automatically detected platform cuda.
INFO 07-10 14:27:08 [core.py:526] Waiting for init message from front-end.
INFO 07-10 14:27:08 [core.py:69] Initializing a V1 LLM engine (v0.9.2) with config: model='/root/autodl-tmp/Qwen2___5-0___5B-Instruct', speculative_config=None, tokenizer='/root/autodl-tmp/Qwen2___5-0___5B-Instruct', skip_tokenizer_init=False, tokenizer_mode=auto, revision=None, override_neuron_config={}, tokenizer_revision=None, trust_remote_code=False, dtype=torch.float16, max_seq_len=2048, download_dir=None, load_format=LoadFormat.AUTO, tensor_parallel_size=1, pipeline_parallel_size=1, disable_custom_all_reduce=False, quantization=None, enforce_eager=True, kv_cache_dtype=auto,  device_config=cuda, decoding_config=DecodingConfig(backend='auto', disable_fallback=False, disable_any_whitespace=False, disable_additional_properties=False, reasoning_backend=''), observability_config=ObservabilityConfig(show_hidden_metrics_for_version=None, otlp_traces_endpoint=None, collect_detailed_traces=None), seed=0, served_model_name=/root/autodl-tmp/Qwen2___5-0___5B-Instruct, num_scheduler_steps=1, multi_step_stream_outputs=True, enable_prefix_caching=True, chunked_prefill_enabled=True, use_async_output_proc=False, pooler_config=None, compilation_config={"level":0,"debug_dump_path":"","cache_dir":"","backend":"","custom_ops":[],"splitting_ops":[],"use_inductor":true,"compile_sizes":[],"inductor_compile_config":{"enable_auto_functionalized_v2":false},"inductor_passes":{},"use_cudagraph":true,"cudagraph_num_of_warmups":0,"cudagraph_capture_sizes":[],"cudagraph_copy_inputs":false,"full_cuda_graph":false,"max_capture_size":0,"local_cache_dir":null}
INFO 07-10 14:27:09 [parallel_state.py:1076] rank 0 in world size 1 is assigned as DP rank 0, PP rank 0, TP rank 0, EP rank 0
WARNING 07-10 14:27:09 [topk_topp_sampler.py:59] FlashInfer is not available. Falling back to the PyTorch-native implementation of top-p & top-k sampling. For the best performance, please install FlashInfer.
INFO 07-10 14:27:09 [gpu_model_runner.py:1770] Starting to load model /root/autodl-tmp/Qwen2___5-0___5B-Instruct...
INFO 07-10 14:27:09 [gpu_model_runner.py:1775] Loading model from scratch...
INFO 07-10 14:27:09 [cuda.py:284] Using Flash Attention backend on V1 engine.
Loading safetensors checkpoint shards:   0% Completed | 0/1 [00:00<?, ?it/s]
Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  3.22it/s]
Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  3.22it/s]

INFO 07-10 14:27:10 [default_loader.py:272] Loading weights took 0.33 seconds
INFO 07-10 14:27:10 [gpu_model_runner.py:1801] Model loading took 0.9277 GiB and 0.467719 seconds
INFO 07-10 14:27:11 [gpu_worker.py:232] Available KV cache memory: 11.71 GiB
INFO 07-10 14:27:11 [kv_cache_utils.py:716] GPU KV cache size: 1,022,896 tokens
INFO 07-10 14:27:11 [kv_cache_utils.py:720] Maximum concurrency for 2,048 tokens per request: 499.46x
INFO 07-10 14:27:11 [core.py:172] init engine (profile, create kv cache, warmup model) took 1.09 seconds

[C] vLLM 批处理（内置 continuous batching）...
Adding requests: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [00:00<00:00, 5333.01it/s]
Processed prompts: 100%|███████████████████████████████████████████████| 50/50 [00:00<00:00, 60.42it/s, est. speed input: 2168.74 toks/s, output: 5514.88 toks/s]
[rank0]:[W710 14:27:13.172024491 ProcessGroupNCCL.cpp:1476] Warning: WARNING: destroy_process_group() was not called before program exit, which can leak resources. For more info, please see https://pytorch.org/docs/stable/distributed.html#shutdown (function operator())

======================================================================
  结果汇总
======================================================================
模式                            总耗时         QPS       tokens/s    相对vLLM    
--------------------------------------------------------------------------------
[A] transformers 串行          55.89s      0.89         82       0.02×
[B] transformers batch=8      9.34s      5.36        486       0.09×
[C] vLLM 批处理                  0.84s     59.60       5438       1.00×

JSON 结果保存：/root/autodl-tmp/vllm_deployment/src/../outputs/throughput_results.json

柱状图已保存：/root/autodl-tmp/vllm_deployment/src/../outputs/throughput_comparison.png

======================================================================
  核心结论：
    vLLM 相对 transformers 串行加速：66.6×
    vLLM 相对 transformers batch:    11.1×
    关键机制：PagedAttention + continuous batching
======================================================================
```
跑完后重新启动 server 继续 demo：
```bash
bash start_server.sh
```

## 四、作为模块调用

除了命令行，也可以把核心逻辑 import 进自己的应用。

### 4.1 启动 server 后用 OpenAI 客户端（推荐）

```python
from openai import OpenAI
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
            "enum": ["营收", "净利润", "ROE", "毛利率", "总资产", "经营现金流","股价"],
        },
    },
    "required": ["company", "year", "metric"],
    "additionalProperties": False,
}
client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")

resp = client.chat.completions.create(
    model="qwen2-0.5b",
    messages=[{"role": "user", "content": "查茅台股价"}],
    extra_body={"guided_json": INTENT_SCHEMA},   # vLLM 扩展字段
    temperature=0,
)
print(resp.choices[0].message.content)
```
**实际运行结果**
```
python python4_1.py 
{"company":"茅台","year":2021,"metric":"总资产"}
```

---

### 4.2 离线批处理（无 server）

```python
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
```
**实际运行结果**
```
python python4_2.py 
INFO 07-10 15:18:13 [__init__.py:244] Automatically detected platform cuda.
INFO 07-10 15:18:19 [config.py:841] This model supports multiple tasks: {'embed', 'classify', 'reward', 'generate'}. Defaulting to 'generate'.
INFO 07-10 15:18:19 [config.py:1472] Using max model len 2048
INFO 07-10 15:18:20 [config.py:2285] Chunked prefill is enabled with max_num_batched_tokens=8192.
INFO 07-10 15:18:20 [core.py:526] Waiting for init message from front-end.
INFO 07-10 15:18:20 [core.py:69] Initializing a V1 LLM engine (v0.9.2) with config: model='/root/autodl-tmp/Qwen2___5-0___5B-Instruct', speculative_config=None, tokenizer='/root/autodl-tmp/Qwen2___5-0___5B-Instruct', skip_tokenizer_init=False, tokenizer_mode=auto, revision=None, override_neuron_config={}, tokenizer_revision=None, trust_remote_code=False, dtype=torch.bfloat16, max_seq_len=2048, download_dir=None, load_format=LoadFormat.AUTO, tensor_parallel_size=1, pipeline_parallel_size=1, disable_custom_all_reduce=False, quantization=None, enforce_eager=False, kv_cache_dtype=auto,  device_config=cuda, decoding_config=DecodingConfig(backend='auto', disable_fallback=False, disable_any_whitespace=False, disable_additional_properties=False, reasoning_backend=''), observability_config=ObservabilityConfig(show_hidden_metrics_for_version=None, otlp_traces_endpoint=None, collect_detailed_traces=None), seed=0, served_model_name=/root/autodl-tmp/Qwen2___5-0___5B-Instruct, num_scheduler_steps=1, multi_step_stream_outputs=True, enable_prefix_caching=True, chunked_prefill_enabled=True, use_async_output_proc=True, pooler_config=None, compilation_config={"level":3,"debug_dump_path":"","cache_dir":"","backend":"","custom_ops":[],"splitting_ops":["vllm.unified_attention","vllm.unified_attention_with_output"],"use_inductor":true,"compile_sizes":[],"inductor_compile_config":{"enable_auto_functionalized_v2":false},"inductor_passes":{},"use_cudagraph":true,"cudagraph_num_of_warmups":1,"cudagraph_capture_sizes":[512,504,496,488,480,472,464,456,448,440,432,424,416,408,400,392,384,376,368,360,352,344,336,328,320,312,304,296,288,280,272,264,256,248,240,232,224,216,208,200,192,184,176,168,160,152,144,136,128,120,112,104,96,88,80,72,64,56,48,40,32,24,16,8,4,2,1],"cudagraph_copy_inputs":false,"full_cuda_graph":false,"max_capture_size":512,"local_cache_dir":null}
INFO 07-10 15:18:21 [parallel_state.py:1076] rank 0 in world size 1 is assigned as DP rank 0, PP rank 0, TP rank 0, EP rank 0
WARNING 07-10 15:18:21 [topk_topp_sampler.py:59] FlashInfer is not available. Falling back to the PyTorch-native implementation of top-p & top-k sampling. For the best performance, please install FlashInfer.
INFO 07-10 15:18:21 [gpu_model_runner.py:1770] Starting to load model /root/autodl-tmp/Qwen2___5-0___5B-Instruct...
INFO 07-10 15:18:21 [gpu_model_runner.py:1775] Loading model from scratch...
INFO 07-10 15:18:21 [cuda.py:284] Using Flash Attention backend on V1 engine.
Loading safetensors checkpoint shards:   0% Completed | 0/1 [00:00<?, ?it/s]
Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  3.55it/s]
Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  3.55it/s]

INFO 07-10 15:18:22 [default_loader.py:272] Loading weights took 0.30 seconds
INFO 07-10 15:18:22 [gpu_model_runner.py:1801] Model loading took 0.9277 GiB and 0.473288 seconds
INFO 07-10 15:18:27 [backends.py:508] Using cache directory: /root/.cache/vllm/torch_compile_cache/5fa0288e5e/rank_0_0/backbone for vLLM's torch.compile
INFO 07-10 15:18:27 [backends.py:519] Dynamo bytecode transform time: 5.27 s
INFO 07-10 15:18:31 [backends.py:155] Directly load the compiled graph(s) for shape None from the cache, took 2.828 s
INFO 07-10 15:18:31 [monitor.py:34] torch.compile takes 5.27 s in total
INFO 07-10 15:18:32 [gpu_worker.py:232] Available KV cache memory: 11.71 GiB
INFO 07-10 15:18:32 [kv_cache_utils.py:716] GPU KV cache size: 1,022,896 tokens
INFO 07-10 15:18:32 [kv_cache_utils.py:720] Maximum concurrency for 2,048 tokens per request: 499.46x
Capturing CUDA graph shapes: 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 67/67 [00:10<00:00,  6.24it/s]
INFO 07-10 15:18:43 [gpu_model_runner.py:2326] Graph capturing finished in 11 secs, took 0.37 GiB
INFO 07-10 15:18:43 [core.py:172] init engine (profile, create kv cache, warmup model) took 20.68 seconds
INFO 07-10 15:18:43 [chat_utils.py:444] Detected the chat template content format to be 'string'. You can set `--chat-template-content-format` to override this.
Adding requests: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:00<00:00, 182.58it/s]
Processed prompts: 100%|████████████████████████████████████████████████████| 2/2 [00:00<00:00,  2.55it/s, est. speed input: 427.46 toks/s, output: 58.69 toks/s]

=== 输出 1 ===
原始文本: {"company": "贵州茅台", "year": 2021, "metric": "总资产"}
解析后数据: {
  "company": "贵州茅台",
  "year": 2021,
  "metric": "总资产"
}
字段校验:
  ✓ company: 贵州茅台
  ✓ year: 2021
  ✓ metric: 总资产

=== 输出 2 ===
原始文本: {"company": "贵州茅台", "year": 2024, "metric": "营收"}
解析后数据: {
  "company": "贵州茅台",
  "year": 2024,
  "metric": "营收"
}
字段校验:
  ✓ company: 贵州茅台
  ✓ year: 2024
  ✓ metric: 营收

```
