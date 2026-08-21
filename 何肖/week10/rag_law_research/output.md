# 解析PDF
```
2026-07-08 10:00:52,349 [INFO]  开始解析: 中华人民共和国刑法.pdf
2026-07-08 10:01:10,311 [INFO]  解析完成: 2254 个块
2026-07-08 10:01:10,362 [INFO]  已保存 → D:\ck\project_info\rag_annual_report\data\parsed\中华人民共和国刑法.json
2026-07-08 10:01:10,363 [INFO]  开始解析: 中华人民共和国宪法.pdf
2026-07-08 10:01:12,037 [INFO]  解析完成: 1036 个块
2026-07-08 10:01:12,058 [INFO]  已保存 → D:\ck\project_info\rag_annual_report\data\parsed\中华人民共和国宪法.json
2026-07-08 10:01:12,059 [INFO]  开始解析: 中华人民共和国民法典.pdf
2026-07-08 10:01:22,955 [INFO]  解析完成: 6934 个块
2026-07-08 10:01:23,077 [INFO]  已保存 → D:\ck\project_info\rag_annual_report\data\parsed\中华人民共和国民法典.json
2026-07-08 10:01:23,078 [INFO]  全部解析完成，结果在 D:\ck\project_info\rag_annual_report\data\parsed
```

# 文档分块
```
2026-07-08 10:04:13,804 [INFO] 分块 中华人民共和国刑法.json  策略=semantic  blocks=2254
2026-07-08 10:04:13,863 [INFO]  → 2165 个 chunk，已保存 中华人民共和国刑法_semantic.json
2026-07-08 10:04:13,869 [INFO] 分块 中华人民共和国宪法.json  策略=semantic  blocks=1036
2026-07-08 10:04:13,893 [INFO]  → 1026 个 chunk，已保存 中华人民共和国宪法_semantic.json
2026-07-08 10:04:13,920 [INFO] 分块 中华人民共和国民法典.json  策略=semantic  blocks=6934
2026-07-08 10:04:14,086 [INFO]  → 6818 个 chunk，已保存 中华人民共和国民法典_semantic.json
2026-07-08 10:04:14,301 [INFO] 合并完成：共 10009 个 chunk → D:\ck\project_info\rag_annual_report\data\chunks\all_semantic.json
2026-07-08 10:04:14,303 [INFO] 平均 chunk 长度: 30 字符
2026-07-08 10:04:14,307 [INFO] 其中表格块: 0  OCR块: 0
```

# 构建索引
```
2026-07-08 10:06:08,530 [INFO] 加载 10009 个 chunks（策略=semantic）
2026-07-08 10:06:09,395 [INFO] Loading faiss with AVX2 support.
2026-07-08 10:06:09,396 [INFO] Could not load library with AVX2 support due to:
ModuleNotFoundError("No module named 'faiss.swigfaiss_avx2'")
2026-07-08 10:06:09,396 [INFO] Loading faiss.
2026-07-08 10:06:09,645 [INFO] Successfully loaded faiss.
2026-07-08 10:06:09,666 [INFO] 开始计算 10009 条 chunk 的 embedding...
2026-07-08 10:06:55,900 [INFO]   Embedding 进度: 100/1001 批
2026-07-08 10:07:41,522 [INFO]   Embedding 进度: 200/1001 批
2026-07-08 10:08:30,164 [INFO]   Embedding 进度: 300/1001 批
2026-07-08 10:09:17,881 [INFO]   Embedding 进度: 400/1001 批
2026-07-08 10:10:03,452 [INFO]   Embedding 进度: 500/1001 批
2026-07-08 10:10:46,873 [INFO]   Embedding 进度: 600/1001 批
2026-07-08 10:11:33,255 [INFO]   Embedding 进度: 700/1001 批
2026-07-08 10:12:21,182 [INFO]   Embedding 进度: 800/1001 批
2026-07-08 10:13:03,954 [INFO]   Embedding 进度: 900/1001 批
2026-07-08 10:13:49,145 [INFO]   Embedding 进度: 1000/1001 批
2026-07-08 10:13:50,764 [INFO]   构建 FAISS 索引，维度=1024...
2026-07-08 10:13:50,796 [INFO]   索引构建完成，共 10009 条向量
2026-07-08 10:13:50,907 [INFO]   FAISS 索引已保存 → D:\ck\project_info\rag_annual_report\vectorstore\faiss_index.bin  (40036 KB)
2026-07-08 10:13:51,138 [INFO]   元数据已保存 → D:\ck\project_info\rag_annual_report\vectorstore\faiss_meta.json
2026-07-08 10:13:51,175 [INFO]   索引构建完成！
2026-07-08 10:13:51,175 [INFO]   FAISS 索引: D:\ck\project_info\rag_annual_report\vectorstore\faiss_index.bin
2026-07-08 10:13:51,175 [INFO]   元数据:     D:\ck\project_info\rag_annual_report\vectorstore\faiss_meta.json
```
# 原生版问答
### 调用 交互
```
2026-07-08 10:37:54,161 [INFO] Loading faiss with AVX2 support.
2026-07-08 10:37:54,162 [INFO] Could not load library with AVX2 support due to:
ModuleNotFoundError("No module named 'faiss.swigfaiss_avx2'")
2026-07-08 10:37:54,162 [INFO] Loading faiss.
2026-07-08 10:37:54,193 [INFO] Successfully loaded faiss.
2026-07-08 10:37:54,308 [INFO] FAISS 索引加载完成，共 10009 条向量
2026-07-08 10:37:54,627 [INFO] 构建 BM25 索引（分词中，请稍候）...
2026-07-08 10:37:54,628 [DEBUG] Building prefix dict from the default dictionary ...
2026-07-08 10:37:54,630 [DEBUG] Loading model from cache C:\Users\nantian\AppData\Local\Temp\jieba.cache
2026-07-08 10:37:55,419 [DEBUG] Loading model cost 0.790 seconds.
2026-07-08 10:37:55,420 [DEBUG] Prefix dict has been built successfully.
2026-07-08 10:37:56,448 [INFO] BM25 索引完成
法律文档 RAG 问答（原生版）
模型：qwen-max  |  向量库：D:\ck\project_info\rag_annual_report\vectorstore\faiss_index.bin
输入 'exit' 退出，'mode' 查看当前配置

问题：醉酒驾驶电动车算危险驾驶罪吗？
2026-07-08 10:38:32,272 [INFO] HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings "HTTP/1.1 200 OK"
2026-07-08 10:38:32,287 [INFO] 向量召回: 10 条，最高分=0.659
2026-07-08 10:38:32,321 [INFO] BM25 召回: 10 条，RRF 后: 18 条
2026-07-08 10:38:56,756 [INFO] No device provided, using cpu
2026-07-08 10:38:56,756 [INFO] No modules.json found for D:\ck\project_info\rag_annual_report\models\bge-reranker-base, initializing a new CrossEncoder model.
Loading weights: 100%|███████████████████████████████████████████████████████████████████████████████████████| 201/201 [00:00<00:00, 4550.19it/s]
Batches: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:04<00:00,  4.02s/it]
2026-07-08 10:39:04,545 [INFO] 最终使用 4 条上下文
2026-07-08 10:39:08,478 [INFO] HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"

============================================================
问题：醉酒驾驶电动车算危险驾驶罪吗？
============================================================

根据提供的资料，第一百三十三条之一规定了危险驾驶罪的情形，其中包括醉酒驾驶机动车[1]。但是，该条款没有明确指出电动车是否属于此处的“机动车”。因此， 仅依据给出的参考资料无法直接确定醉酒驾驶电动车是否构成危险驾驶罪。需要参考其他相关法律法规或司法解释来进一步明确电动车在特定情况下的法律地位。根据提供的资料无法回答此问题。

── 来源 ──
  [1] 中华人民共和国刑法 · 第二编　分　　则 > 第二章　危害公共安全罪 > 第一百一十四条　放火、决水、爆炸以及投放毒害性、放射性、传染病 > 第一百三十三条之一　在道路上驾驶机动车，有下列情形之一的，处拘 · 第29页
  [2] 中华人民共和国刑法 · 第二编　分　　则 > 第二章　危害公共安全罪 > 第一百一十四条　放火、决水、爆炸、投毒或者以其他危险方法破坏工 > 第一百一十六条　破坏火车、汽车、电车、船只、航空器，足以使火 · 第139页
  [3] 中华人民共和国刑法 · 第二编　分　　则 > 第二章　危害公共安全罪 > 第一百一十四条　放火、决水、爆炸以及投放毒害性、放射性、传染病 > 第一百一十六条　破坏火车、汽车、电车、船只、航空器，足以使火 · 第25页
  [4] 中华人民共和国民法典 · 第七编 > 第五章 > 机动车交通事故责任 · 第236页
  
问题：在微信群里骂人犯法吗？侵犯了什么权利？
2026-07-08 10:40:20,260 [INFO] HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings "HTTP/1.1 200 OK"
2026-07-08 10:40:20,272 [INFO] 向量召回: 10 条，最高分=0.643
2026-07-08 10:40:20,334 [INFO] BM25 召回: 10 条，RRF 后: 20 条
2026-07-08 10:40:20,342 [INFO] No device provided, using cpu
2026-07-08 10:40:20,343 [INFO] No modules.json found for D:\ck\project_info\rag_annual_report\models\bge-reranker-base, initializing a new CrossEncoder model.
Loading weights: 100%|███████████████████████████████████████████████████████████████████████████████████████| 201/201 [00:00<00:00, 4129.34it/s]
Batches: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:01<00:00,  1.44s/it]
2026-07-08 10:40:25,501 [INFO] 最终使用 4 条上下文
2026-07-08 10:40:29,169 [INFO] HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"

============================================================
问题：在微信群里骂人犯法吗？侵犯了什么权利？
============================================================

在微信群里骂人可能构成违法，具体来说，如果使用了侮辱性语言或捏造事实诽谤他人，则侵犯了他人的名誉权[1]。根据《中华人民共和国刑法》第二百四十六条的 规定，以暴力或者其他方法公然侮辱他人或者捏造事实诽谤他人的行为是被禁止的[2][3]。因此，在微信群这样的公开平台上辱骂他人不仅违反了民法中关于保护个人名誉权的规定，情节严重时还可能触犯刑法。

── 来源 ──
  [1] 中华人民共和国民法典 · 第四编 > 第五章 > 名誉权和荣誉权 > 人不得以侮辱、诽谤等方式侵害他人的名誉权。 · 第198页
  [2] 中华人民共和国刑法 · 第二编　分　　则 > 第四章　侵犯公民人身权利、民主权利罪 > 第二百三十二条　故意杀人的，处死刑、无期徒刑或者十年以上有期 徒 > 第二百四十六条　以暴力或者其他方法公然侮辱他人或者捏造事实诽谤 · 第172页
  [3] 中华人民共和国刑法 · 第二编　分　　则 > 第四章　侵犯公民人身权利、民主权利罪 > 第二百三十二条　故意杀人的，处死刑、无期徒刑或者十年以上有期 徒 > 第二百四十六条　以暴力或者其他方法公然侮辱他人或者捏造事实诽谤 · 第66页
  [4] 中华人民共和国民法典 · 第七编 > 第二章 > 损害赔偿 > 侵害自然人人身权益造成严重精神 · 第230页

```

### 调用 单次查询
```
(base) PS D:\ck\project_info\rag_annual_report\src> python .\rag_pipeline.py --query "有人欠钱不还，我能拿欠钱人的东西抵押吗？"
2026-07-08 10:47:19,154 [INFO] Loading faiss with AVX2 support.
2026-07-08 10:47:19,155 [INFO] Could not load library with AVX2 support due to:ModuleNotFoundError("No module named 'faiss.swigfaiss_avx2'")
2026-07-08 10:47:19,156 [INFO] Loading faiss.
2026-07-08 10:47:19,190 [INFO] Successfully loaded faiss.
2026-07-08 10:47:19,298 [INFO] FAISS 索引加载完成，共 10009 条向量
2026-07-08 10:47:19,686 [INFO] 构建 BM25 索引（分词中，请稍候）...
2026-07-08 10:47:19,687 [DEBUG] Building prefix dict from the default dictionary ...
2026-07-08 10:47:19,689 [DEBUG] Loading model from cache C:\Users\nantian\AppData\Local\Temp\jieba.cache
2026-07-08 10:47:20,572 [DEBUG] Loading model cost 0.884 seconds.
2026-07-08 10:47:20,572 [DEBUG] Prefix dict has been built successfully.
2026-07-08 10:47:21,734 [INFO] BM25 索引完成
2026-07-08 10:47:23,386 [INFO] HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings "HTTP/1.1 200 OK"
2026-07-08 10:47:23,401 [INFO] 向量召回: 10 条，最高分=0.809
2026-07-08 10:47:23,449 [INFO] BM25 召回: 10 条，RRF 后: 20 条
2026-07-08 10:47:41,220 [INFO] No device provided, using cpu
2026-07-08 10:47:41,221 [INFO] No modules.json found for D:\ck\project_info\rag_annual_report\models\bge-reranker-base, initializing a new CrossEncoder model.
Loading weights: 100%|███████████████████████████████████████████████████████████████████████████████████████| 201/201 [00:00<00:00, 4781.93it/s]
Batches: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:01<00:00,  1.72s/it]
2026-07-08 10:47:45,821 [INFO] 最终使用 4 条上下文
2026-07-08 10:47:48,269 [INFO] HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"

============================================================
问题：有人欠钱不还，我能拿欠钱人的东西抵押吗？
============================================================

根据提供的资料无法回答此问题。参考资料中没有直接提到在个人之间债务不履行的情况下，是否可以直接拿欠钱人的东西作为抵押。但根据《中华人民共和国民法典》的规定，可以抵押的财产需要符合法律规定[2]，并且关于抵押权的实现也有特定的要求[4]。因此，在实际操作前，建议咨询法律专业人士或通过合法途径解决债务纠纷。

── 来源 ──
  [1] 中华人民共和国民法典 · 第二编 > 第十七章 > 第一节 > 但是法律规定可以抵押的除外； · 第79页
  [2] 中华人民共和国民法典 · 第二编 > 第十七章 > 第一节 > 可以抵押： · 第78页
  [3] 中华人民共和国民法典 · 第二编 > 第十七章 > 第一节 > 约定债务人不履行到期债务时抵押财产归债权人所有的，只能依 · 第80页
  [4] 中华人民共和国民法典 · 第二编 > 第十七章 > 第一节 > 定的实现抵押权的情形，抵押权人可以与抵押人协议以抵押财产 · 第82页

```
### 调用 加过滤条件（只检索特定法律）
```
(base) PS D:\ck\project_info\rag_annual_report\src> python .\rag_pipeline.py --query "小区物业能停水停电催缴物业费吗？" --raw_name "中华人民共和国民法典" " 
2026-07-08 10:52:30,421 [INFO] Loading faiss with AVX2 support.
2026-07-08 10:52:30,422 [INFO] Could not load library with AVX2 support due to:ModuleNotFoundError("No module named 'faiss.swigfaiss_avx2'")
2026-07-08 10:52:30,423 [INFO] Loading faiss.
2026-07-08 10:52:30,493 [INFO] Successfully loaded faiss.
2026-07-08 10:52:30,602 [INFO] FAISS 索引加载完成，共 10009 条向量
2026-07-08 10:52:30,929 [INFO] 构建 BM25 索引（分词中，请稍候）...
2026-07-08 10:52:30,929 [DEBUG] Building prefix dict from the default dictionary ...
2026-07-08 10:52:30,931 [DEBUG] Loading model from cache C:\Users\nantian\AppData\Local\Temp\jieba.cache
2026-07-08 10:52:31,671 [DEBUG] Loading model cost 0.741 seconds.
2026-07-08 10:52:31,672 [DEBUG] Prefix dict has been built successfully.
2026-07-08 10:52:32,712 [INFO] BM25 索引完成
2026-07-08 10:52:34,175 [INFO] HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings "HTTP/1.1 200 OK"
2026-07-08 10:52:34,185 [INFO] 向量召回: 10 条，最高分=0.835
2026-07-08 10:52:34,222 [INFO] BM25 召回: 10 条，RRF 后: 16 条
2026-07-08 10:52:51,589 [INFO] No device provided, using cpu
2026-07-08 10:52:51,590 [INFO] No modules.json found for D:\ck\project_info\rag_annual_report\models\bge-reranker-base, initializing a new CrossEncoder model.
Loading weights: 100%|███████████████████████████████████████████████████████████████████████████████████████| 201/201 [00:00<00:00, 4173.50it/s]
Batches: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:01<00:00,  1.02s/it]
2026-07-08 10:52:55,459 [INFO] 最终使用 4 条上下文
2026-07-08 10:53:02,338 [INFO] HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"

============================================================
问题：小区物业能停水停电催缴物业费吗？
============================================================

根据中华人民共和国民法典的规定，物业服务人不得采取停止供电、供水、供热、供燃气等方式催缴物业费[3]。因此，小区物业不能通过停水停电的方式来催缴物业 费。

── 来源 ──
  [1] 中华人民共和国民法典 · 第三编 > 第二十四章 > 物业服务合同 > 催交物业费。 · 第183页
  [2] 中华人民共和国民法典 · 第三编 > 第二十四章 > 物业服务合同 > 业主违反约定逾期不支付物业费的，物业服务人可以催告其 · 第183页
  [3] 中华人民共和国民法典 · 第三编 > 第二十四章 > 物业服务合同 > 物业服务人不得采取停止供电、供水、供热、供燃气等方式 · 第183页
  [4] 中华人民共和国民法典 · 第三编 > 第二十四章 > 物业服务合同 > 的物业费。 · 第184页
```
### 调用 开启查询改写
```
(base) PS D:\ck\project_info\rag_annual_report\src> python .\rag_pipeline.py --query "遗嘱" --query-rewrite
2026-07-08 10:56:09,776 [INFO] Loading faiss with AVX2 support.
2026-07-08 10:56:09,777 [INFO] Could not load library with AVX2 support due to:ModuleNotFoundError("No module named 'faiss.swigfaiss_avx2'")
2026-07-08 10:56:09,778 [INFO] Loading faiss.
2026-07-08 10:56:09,817 [INFO] Successfully loaded faiss.
2026-07-08 10:56:09,946 [INFO] FAISS 索引加载完成，共 10009 条向量
2026-07-08 10:56:10,290 [INFO] 构建 BM25 索引（分词中，请稍候）...
2026-07-08 10:56:10,291 [DEBUG] Building prefix dict from the default dictionary ...
2026-07-08 10:56:10,294 [DEBUG] Loading model from cache C:\Users\nantian\AppData\Local\Temp\jieba.cache
2026-07-08 10:56:11,025 [DEBUG] Loading model cost 0.734 seconds.
2026-07-08 10:56:11,025 [DEBUG] Prefix dict has been built successfully.
2026-07-08 10:56:12,013 [INFO] BM25 索引完成
2026-07-08 10:56:13,520 [INFO] HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
2026-07-08 10:56:13,537 [INFO] 查询改写: '遗嘱' → '遗嘱效力认定及法律依据'
2026-07-08 10:56:13,724 [INFO] HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings "HTTP/1.1 200 OK"
2026-07-08 10:56:13,736 [INFO] 向量召回: 10 条，最高分=0.785
2026-07-08 10:56:13,762 [INFO] BM25 召回: 10 条，RRF 后: 19 条
2026-07-08 10:56:31,433 [INFO] No device provided, using cpu
2026-07-08 10:56:31,434 [INFO] No modules.json found for D:\ck\project_info\rag_annual_report\models\bge-reranker-base, initializing a new CrossEncoder model.
Loading weights: 100%|███████████████████████████████████████████████████████████████████████████████████████| 201/201 [00:00<00:00, 4688.61it/s]
Batches: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:01<00:00,  1.01s/it]
2026-07-08 10:56:35,227 [INFO] 最终使用 4 条上下文
2026-07-08 10:56:38,915 [INFO] HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"

============================================================
问题：遗嘱
============================================================

遗嘱是遗嘱人生前依法处分其个人财产及安排相关事务，并于死亡时发生效力的法律行为。根据参考资料，遗嘱必须反映遗嘱人的真实意愿，如果遗嘱是在受到欺诈或胁迫的情况下订立的，则该遗嘱无效[3]。此外，对于自书遗嘱而言，要求由遗嘱人亲自书写并签名[4]。这些规定旨在确保遗嘱能够准确体现遗嘱人的最终意愿。

── 来源 ──
  [1] 中华人民共和国民法典 · 第六编 > 第三章 > 遗嘱继承和遗赠 · 第222页
  [2] 中华人民共和国民法典 · 第六编 > 第三章 > 遗嘱继承和遗赠 · 第6页
  [3] 中华人民共和国民法典 · 第六编 > 第三章 > 遗嘱继承和遗赠 > 遗嘱必须表示遗嘱人的真实意思，受欺诈、胁迫所立的遗嘱 · 第223页
  [4] 中华人民共和国民法典 · 第六编 > 第三章 > 遗嘱继承和遗赠 > 自书遗嘱由遗嘱人亲笔书写，签名， · 第222页
```
### 调用 消融测试：关闭 BM25 或 Rerank
```
(base) PS D:\ck\project_info\rag_annual_report\src> python .\rag_pipeline.py --query "公司辞退我，我能拿赔偿金吗？依据是什么？" --no-bm25 
2026-07-08 11:01:56,780 [INFO] Loading faiss with AVX2 support.
2026-07-08 11:01:56,781 [INFO] Could not load library with AVX2 support due to:ModuleNotFoundError("No module named 'faiss.swigfaiss_avx2'")
2026-07-08 11:01:56,781 [INFO] Loading faiss.
2026-07-08 11:01:56,819 [INFO] Successfully loaded faiss.
2026-07-08 11:01:56,960 [INFO] FAISS 索引加载完成，共 10009 条向量
2026-07-08 11:01:58,772 [INFO] HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings "HTTP/1.1 200 OK"
2026-07-08 11:01:58,783 [INFO] 向量召回: 10 条，最高分=0.712
2026-07-08 11:02:21,301 [INFO] No device provided, using cpu
2026-07-08 11:02:21,302 [INFO] No modules.json found for D:\ck\project_info\rag_annual_report\models\bge-reranker-base, initializing a new CrossEncoder model.
Loading weights: 100%|███████████████████████████████████████████████████████████████████████████████████████| 201/201 [00:00<00:00, 3424.02it/s]
Batches: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  1.73it/s]
2026-07-08 11:02:25,446 [INFO] 最终使用 4 条上下文
2026-07-08 11:02:27,827 [INFO] HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"

============================================================
问题：公司辞退我，我能拿赔偿金吗？依据是什么？
============================================================

根据提供的资料无法回答此问题。参考资料中没有直接提到关于公司辞退员工时赔偿金的具体规定。您的情况可能涉及到劳动合同法或其他相关法律法规，但这些内容并未在给出的参考资料中出现。建议查阅《中华人民共和国劳动合同法》等相关法律条文或咨询专业律师以获得准确答案。

── 来源 ──
  [1] 中华人民共和国民法典 · 第二编 > 第十六章 > 一般规定 > 赔偿金或者补偿金等。 · 第77页
  [2] 中华人民共和国民法典 · 第二编 > 第八章 > 共有 > 的，应当给予赔偿。 · 第63页
  [3] 中华人民共和国民法典 · 第七编 > 第四章 > 产品责任 > 赔偿。 · 第235页
  [4] 中华人民共和国宪法 · －1－ > 第二章 > 公民的基本权利和义务 > 人，有依照法律规定取得赔偿的权利。 · 第15页

(base) PS D:\ck\project_info\rag_annual_report\src> python .\rag_pipeline.py --query "公司辞退我，我能拿赔偿金吗？依据是什么？" --no-rerank 
2026-07-08 11:03:39,600 [INFO] Loading faiss with AVX2 support.
2026-07-08 11:03:39,601 [INFO] Could not load library with AVX2 support due to:ModuleNotFoundError("No module named 'faiss.swigfaiss_avx2'")
2026-07-08 11:03:39,601 [INFO] Loading faiss.
2026-07-08 11:03:39,621 [INFO] Successfully loaded faiss.
2026-07-08 11:03:39,712 [INFO] FAISS 索引加载完成，共 10009 条向量
2026-07-08 11:03:40,043 [INFO] 构建 BM25 索引（分词中，请稍候）...
2026-07-08 11:03:40,044 [DEBUG] Building prefix dict from the default dictionary ...
2026-07-08 11:03:40,046 [DEBUG] Loading model from cache C:\Users\nantian\AppData\Local\Temp\jieba.cache
2026-07-08 11:03:40,780 [DEBUG] Loading model cost 0.736 seconds.
2026-07-08 11:03:40,781 [DEBUG] Prefix dict has been built successfully.
2026-07-08 11:03:41,646 [INFO] BM25 索引完成
2026-07-08 11:03:42,738 [INFO] HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings "HTTP/1.1 200 OK"
2026-07-08 11:03:42,745 [INFO] 向量召回: 10 条，最高分=0.712
2026-07-08 11:03:42,771 [INFO] BM25 召回: 10 条，RRF 后: 19 条
2026-07-08 11:03:42,771 [INFO] 最终使用 4 条上下文
2026-07-08 11:03:46,078 [INFO] HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"

============================================================
问题：公司辞退我，我能拿赔偿金吗？依据是什么？
============================================================

根据提供的资料无法回答此问题。参考资料中没有直接提到关于公司辞退员工时赔偿金的相关规定。您的问题可能涉及到劳动合同法或其他相关法律法规，但这些内容并未包含在给出的参考资料内。

── 来源 ──
  [1] 中华人民共和国民法典 · 第二编 > 第十六章 > 一般规定 > 赔偿金或者补偿金等。 · 第77页
  [2] 中华人民共和国民法典 · 第七编 > 第四章 > 产品责任 > 赔偿。 · 第235页
  [3] 中华人民共和国民法典 · 第三编 > 第十八章 > 建设工程合同 > 并有权请求赔偿停工、窝工等损失。 · 第155页
  [4] 中华人民共和国民法典 · 第七编 > 第二章 > 损害赔偿 > 葬费和死亡赔偿金。 · 第230页

```

# LangChain 版
### 构建 LangChain 向量索引
```
2026-07-08 13:40:25,870 [INFO] 加载: 中华人民共和国刑法.pdf
2026-07-08 13:40:26,589 [INFO]   → 285 页
2026-07-08 13:40:26,589 [INFO] 加载: 中华人民共和国宪法.pdf
2026-07-08 13:40:26,628 [INFO]   → 40 页
2026-07-08 13:40:26,628 [INFO] 加载: 中华人民共和国民法典.pdf
2026-07-08 13:40:26,769 [INFO]   → 246 页
2026-07-08 13:40:26,769 [INFO] 共加载 571 页（来自 3 个文件）
2026-07-08 13:40:26,801 [INFO] 分块完成：571 页 → 2323 个 chunk
2026-07-08 13:40:26,801 [INFO] 平均 chunk 长度：143 字符
2026-07-08 13:40:26,876 [INFO] Loading SentenceTransformer model from D:\ck\project_info\rag_annual_report\models\bge-small-zh-v1.5.
Loading weights: 100%|██████████| 71/71 [00:00<00:00, 4176.36it/s]
2026-07-08 13:40:27,192 [INFO] Embedding 模型加载完成: D:\ck\project_info\rag_annual_report\models\bge-small-zh-v1.5
2026-07-08 13:40:27,208 [INFO] 构建向量库（2323 个 chunk）...
2026-07-08 13:41:00,428 [INFO] Loading faiss with AVX2 support.
2026-07-08 13:41:00,429 [INFO] Could not load library with AVX2 support due to:ModuleNotFoundError("No module named 'faiss.swigfaiss_avx2'")
2026-07-08 13:41:00,429 [INFO] Loading faiss.
2026-07-08 13:41:00,992 [INFO] Successfully loaded faiss.

LangChain 向量库构建完成！
  路径: D:\ck\project_info\rag_annual_report\vectorstore\faiss_lc
  下一步: python src_langchain/rag_chain_lc.py
2026-07-08 13:41:01,064 [INFO] 向量库已保存 → D:\ck\project_info\rag_annual_report\vectorstore\faiss_lc
2026-07-08 13:41:01,065 [INFO]   index.faiss: 4646 KB
```
# LangChain 版问答
### 交互式
```
(base) PS D:\ck\project_info\rag_annual_report\src_langchain> python .\rag_chain_lc.py            
2026-07-08 13:52:00,961 [INFO] 加载 embedding 模型...
2026-07-08 13:53:02,600 [INFO] Loading SentenceTransformer model from D:\ck\project_info\rag_annual_report\models\bge-small-zh-v1.5.
Loading weights: 100%|█████████████████████████████████████████████████████████████████████████████████████████| 71/71 [00:00<00:00, 5758.29it/s]
2026-07-08 13:53:02,860 [INFO] 加载向量库...
D:\ck\project_info\rag_annual_report\src_langchain\rag_chain_lc.py:94: DeprecationWarning: `langchain-community` is being sunset and is no longer actively maintained. See https://github.com/langchain-ai/langchain-community/issues/674 for details and migration guidance toward standalone integration packages.
  from langchain_community.vectorstores import FAISS
2026-07-08 13:53:02,900 [INFO] Loading faiss with AVX2 support.
2026-07-08 13:53:02,901 [INFO] Could not load library with AVX2 support due to:
ModuleNotFoundError("No module named 'faiss.swigfaiss_avx2'")
2026-07-08 13:53:02,901 [INFO] Loading faiss.
2026-07-08 13:53:03,536 [INFO] Successfully loaded faiss.
法律文档 RAG 问答系统（LangChain LCEL 版）
模型：qwen-plus  |  向量库：D:\ck\project_info\rag_annual_report\vectorstore\faiss_lc
输入 'exit' 退出

问题：未成年人偷东西犯法吗?   

============================================================
问题：未成年人偷东西犯法吗?
============================================================
2026-07-08 13:53:58,293 [INFO] HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"

是的，未成年人偷东西可能构成犯罪，是否负刑事责任取决于其年龄和行为性质：

- 已满十六周岁的人盗窃，应当负刑事责任（来源：中华人民共和国刑法.pdf 第6页、第122页）；  
- 已满十四周岁不满十六周岁的人，**盗窃罪不在其应当负刑事责任的八种特定严重犯罪之列**（该年龄段仅对故意杀人、故意伤害致人重伤或死亡、强奸、抢劫、贩卖毒品、放火、爆炸、投毒罪负刑事责任）（来源：中华人民共和国刑法.pdf 第6页、第122页）；  
- 但若组织未成年人进行盗窃活动，则构成《刑法》第二百六十二条之二规定的犯罪，组织者无论自身年龄如何，均应承担刑事责任（来源：中华人民共和国刑法.pdf 第69页）。

综上，未成年人本人实施盗窃行为：  
→ 满16周岁：构成盗窃罪，应负刑事责任；  
→ 14至16周岁：一般不因盗窃负刑事责任（除非转化为抢劫等严重情形）；  
→ 不满14周岁：不负刑事责任，但可能依法予以矫治教育。

注：参考资料中未载明盗窃罪的具体条文（如第二百六十四条），故无法援引具体量刑规定，仅依据现有资料作上述判断。
问题：我把人打伤了，要赔多少钱？  

============================================================
问题：我把人打伤了，要赔多少钱？
============================================================
2026-07-08 13:54:33,214 [INFO] HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"

根据【参考资料】，您将承担民事赔偿和可能的刑事责任，具体金额需结合实际情况确定：

一、民事赔偿（依据《民法典》）：  
若您侵害他人造成人身损害，应赔偿：  
- 医疗费、护理费、交通费、营养费、住院伙食补助费等为治疗和康复支出的合理费用；  
- 因误工减少的收入；  
- 若造成残疾，还应赔偿辅助器具费和残疾赔偿金；  
- 若造成死亡，还应赔偿丧葬费和死亡赔偿金（来源：[1] 第一千一百七十九条）。  

若人身权益受损同时造成财产损失，按被侵权人实际损失或侵权人获利赔偿；难以确定时，由法院根据实际情况判定（来源：[2] 第一千一百八十二条）。  
若造成严重精神损害，还需赔偿精神损害抚慰金（来源：[2] 第一千一百八十三条）。

二、刑事责任（依据《刑法》）：  
故意伤害他人身体的，处三年以下有期徒刑、拘役或者管制；致人重伤的，处三年以上十年以下有期徒刑；致人死亡或以特别残忍手段致人重伤造成严重残疾的，处十年以上有期徒刑、无期徒刑或死刑（来源：[4] 第二百三十四条）。

⚠️注意：赔偿金额无法笼统确定，须根据伤情程度、实际支出、收入状况、当地标准及法院裁量综合认定。是否构成犯罪亦取决于伤情鉴定结果。

综上，具体赔偿数额需依个案事实和证据，由双方协商或人民法院依法判决确定。
问题：
```
### 单次查询（通过 stdin）
```
(base) PS D:\ck\project_info\rag_annual_report\src_langchain> echo "故意伤害致人轻伤会坐牢吗" | python .\rag_chain_lc.py                          
2026-07-08 14:04:44,019 [INFO] 加载 embedding 模型...                                                                                             
2026-07-08 14:04:56,220 [INFO] Loading SentenceTransformer model from D:\ck\project_info\rag_annual_report\models\bge-small-zh-v1.5.              
Loading weights: 100%|█████████████████████████████████████████████████████████████████████████████████████████| 71/71 [00:00<00:00, 8032.46it/s] 
2026-07-08 14:04:56,344 [INFO] 加载向量库...                                                                                                      
D:\ck\project_info\rag_annual_report\src_langchain\rag_chain_lc.py:98: DeprecationWarning: `langchain-community` is being sunset and is no longer actively maintained. See https://github.com/langchain-ai/langchain-community/issues/674 for details and migration guidance toward standalone integration packages.                                                                                                                                  
  from langchain_community.vectorstores import FAISS                                                                                              
2026-07-08 14:04:56,371 [INFO] Loading faiss with AVX2 support.                                                                                   
2026-07-08 14:04:56,372 [INFO] Could not load library with AVX2 support due to:                                                                   
ModuleNotFoundError("No module named 'faiss.swigfaiss_avx2'")                                                                                     
2026-07-08 14:04:56,372 [INFO] Loading faiss.                                                                                                     
2026-07-08 14:04:56,395 [INFO] Successfully loaded faiss.                                                                                         
法律文档 RAG 问答系统（LangChain LCEL 版）                                                                                                        
模型：qwen-plus  |  向量库：D:\ck\project_info\rag_annual_report\vectorstore\faiss_lc                                                             
输入 'exit' 退出                                                                                                                                  
                                                                                                                                                  
问题：2026-07-08 14:05:01,560 [INFO] HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"      
                                                                                                                                                  
============================================================
问题：故意伤害致人轻伤会坐牢吗
============================================================

根据【参考资料】[1]和[4]，故意伤害他人身体的，处三年以下有期徒刑、拘役或者管制（来源：中华人民共和国刑法，第62页、第169页）。  
“轻伤”属于“故意伤害他人身体”的基本情形（未达重伤标准），依法应处三年以下有期徒刑、拘役或者管制。  
是否实际判处实刑（即“坐牢”），需结合案件具体情节、悔罪表现、赔偿谅解等因素由法院裁量；但该行为已构成犯罪，存在被判处有期徒刑的可能性。

因此，**故意伤害致人轻伤可能坐牢，法定刑为三年以下有期徒刑、拘役或者管制**（来源：中华人民共和国刑法，第62页、第169页）。
```

# 在代码中直接调用（作为模块使用）
```
2026-07-08 14:13:17,968 [INFO] Loading faiss with AVX2 support.
2026-07-08 14:13:17,969 [INFO] Could not load library with AVX2 support due to:
ModuleNotFoundError("No module named 'faiss.swigfaiss_avx2'")
2026-07-08 14:13:17,969 [INFO] Loading faiss.
2026-07-08 14:13:17,995 [INFO] Successfully loaded faiss.
2026-07-08 14:13:18,106 [INFO] FAISS 索引加载完成，共 10009 条向量
C:\Users\nantian\AppData\Local\Programs\Python\Python312\Lib\site-packages\jieba\_compat.py:18: UserWarning: pkg_resources is deprecated as an API. See https://setuptools.pypa.io/en/latest/pkg_resources.html. The pkg_resources package is slated for removal as early as 2025-11-30. Refrain from using this package or pin to Setuptools<81.
  import pkg_resources
2026-07-08 14:13:18,315 [INFO] 构建 BM25 索引（分词中，请稍候）...
2026-07-08 14:13:18,315 [DEBUG] Building prefix dict from the default dictionary ...
2026-07-08 14:13:18,317 [DEBUG] Loading model from cache C:\Users\nantian\AppData\Local\Temp\jieba.cache
2026-07-08 14:13:18,772 [DEBUG] Loading model cost 0.457 seconds.
2026-07-08 14:13:18,772 [DEBUG] Prefix dict has been built successfully.
2026-07-08 14:13:19,284 [INFO] BM25 索引完成
2026-07-08 14:13:20,412 [INFO] HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings "HTTP/1.1 200 OK"
2026-07-08 14:13:22,328 [INFO] HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
不可以，政府征用土地必须是为了公共利益的需要，并且要依照法律规定进行[3]。此外，法律规定的权限和程序可以征用组织、个人的不动产或者动产，这意味着征用行为并非随意，而是需要遵循特定的法律程序[1]。
[{'index': 1, 'source': '[1] 中华人民共和国民法典 · 第二编 > 第四章 > 一般规定 > 法律规定的权限和程序可以征用组织、个人的不动产或者动产。 · 第52页', 'chunk_id': '中华人民共和国民法典_01493'}, {'index': 2, 'source': '[2] 中华人民共和国民法典 · 第二编 > 第十二章 > 建设用地使用权 > 建设用地使用权可以在土地的地表、地上 · 第69页', 'chunk_id': '中华人民共和国民法典_01979'}, {'index': 3, 'source': '[3] 中华人民共和国宪法 · －1－ > 第一章 > －6－ > 国家为了公共利益的需要，可以依照法律规定对土地实行征 · 第8页', 'chunk_id': '中华人民共和国宪法_00191'}, {'index': 4, 'source': '[4] 中华人民共和国宪法 · －1－ > 第一章 > －6－ > 土地。土地的使用权可以依照法律的规定转让。 · 第9页', 'chunk_id': '中华人民共和国宪法_00195'}]

```
# 调用 LangChain 版 Chain
```
2026-07-08 14:18:54,387 [INFO] Loading SentenceTransformer model from D:\ck\project_info\rag_annual_report\models\bge-small-zh-v1.5.
Loading weights: 100%|██████████| 71/71 [00:00<00:00, 8877.50it/s]
2026-07-08 14:18:54,518 [INFO] Loading faiss with AVX2 support.
2026-07-08 14:18:54,518 [INFO] Could not load library with AVX2 support due to:
ModuleNotFoundError("No module named 'faiss.swigfaiss_avx2'")
2026-07-08 14:18:54,518 [INFO] Loading faiss.
2026-07-08 14:18:54,542 [INFO] Successfully loaded faiss.
2026-07-08 14:18:59,009 [INFO] HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
根据提供的参考资料，仅能确认《中华人民共和国宪法》规定了“公民的基本权利和义务”属于第二章内容（来源：[4] 中华人民共和国宪法.pdf，目录），但【参考资料】中未具体列出公民有哪些基本权利（如选举权、人身自由、言论自由等具体内容未在所给页码或片段中出现）。

因此，**根据提供的资料无法回答此问题**。
['。\n中华人民共和国公民在法律面前一律平等。\n国家尊重和保障人权。\n任何公民享有宪法和法律规定的权利，同时必须履行宪法和', '。\n在国家机关、国有公司、企业、集体企业和人民团体管理、使用或者运\n输中的私人财产，以公共财产论。\n第九十二条\u3000本法所称公民私人所有的财产，是指下列财产：\n（一）公民的合法收入、储蓄、房屋和其他生活资料；\n（二）依法归个人、家庭所有的生产资料；\n（三）个体户和私营企业的合法财产；\n（四）依法归个人所有的股份、股票、债券和其他财产。\n第九十三条\u3000本法所称国家工作人员，是指国家机关中从事公务的人\n员', '。\n在国家机关、国有公司、企业、集体企业和人民团体管理、使用或者运\n输中的私人财产，以公共财产论。\n第九十二条\u3000本法所称公民私人所有的财产，是指下列财产：\n（一）公民的合法收入、储蓄、房屋和其他生活资料；\n（二）依法归个人、家庭所有的生产资料；\n（三）个体户和私营企业的合法财产；\n（四）依法归个人所有的股份、股票、债券和其他财产。\n第九十三条\u3000本法所称国家工作人员，是指国家机关中从事公务的人\n员', '目\n录\n序\n言\n第一章\n总\n纲\n第二章\n公民的基本权利和义务\n第三章\n国家机构']

```
# HTTP 服务模式
```
INFO:     Started server process [32816]
INFO:     Waiting for application startup.
2026-07-08 15:17:03,104 [INFO] 服务启动，初始化 RAG Pipeline...
2026-07-08 15:17:04,820 [INFO] Loading faiss with AVX2 support.
2026-07-08 15:17:04,821 [INFO] Could not load library with AVX2 support due to:ModuleNotFoundError("No module named 'faiss.swigfaiss_avx2'")
2026-07-08 15:17:04,821 [INFO] Loading faiss.
2026-07-08 15:17:04,844 [INFO] Successfully loaded faiss.
2026-07-08 15:17:04,929 [INFO] FAISS 索引加载完成，共 10009 条向量
2026-07-08 15:17:05,196 [INFO] 构建 BM25 索引（分词中，请稍候）...
2026-07-08 15:17:05,197 [DEBUG] Building prefix dict from the default dictionary ...
2026-07-08 15:17:05,199 [DEBUG] Loading model from cache C:\Users\nantian\AppData\Local\Temp\jieba.cache
2026-07-08 15:17:05,937 [DEBUG] Loading model cost 0.740 seconds.
2026-07-08 15:17:05,938 [DEBUG] Prefix dict has been built successfully.
2026-07-08 15:17:06,946 [INFO] BM25 索引完成
2026-07-08 15:17:06,948 [INFO] Pipeline 初始化完成，开始接受请求
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     127.0.0.1:52628 - "GET / HTTP/1.1" 200 OK
```
![1.png](Http_%E6%9C%8D%E5%8A%A1/1.png)
![2.png](Http_%E6%9C%8D%E5%8A%A1/2.png)
![3.png](Http_%E6%9C%8D%E5%8A%A1/3.png)
![4.png](Http_%E6%9C%8D%E5%8A%A1/4.png)


# 接口测试
![1.png](%E6%8E%A5%E5%8F%A3%E6%B5%8B%E8%AF%95/1.png)
![2.png](%E6%8E%A5%E5%8F%A3%E6%B5%8B%E8%AF%95/2.png)
![3.png](%E6%8E%A5%E5%8F%A3%E6%B5%8B%E8%AF%95/3.png)

# 接口调用
### Python requests：
```python
import requests

resp = requests.post(
    "http://localhost:8000/query",
    json={"question": "我被前公司领导性骚扰，已经离职了还能告他吗？"},
)
data = resp.json()
print(data["answer"])
for c in data["citations"]:
    print(f"  [{c['index']}] {c['source']}")
```
### 调用结果
```
根据提供的资料，中华人民共和国民法典中提到，利用职权、从属关系等实施性骚扰的行为应当被防止和制止，并且受害人有权依法请求行为人承担民事责任[3]。这意味着即使你已经离职，仍然可以就之前遭受的性骚扰行为向法院提起诉讼，要求前公司领导承担相应的法律责任。因此，答案是可以的，你依然能够对他提起诉讼。不过，具体的法律程序和所需证据建议咨询专业律师以获得更详细的指导。
  [1] [1] 中华人民共和国民法典 · 第四编 > 第二章 > 生命权、身体权和健康权 > 体行为等方式对他人实施性骚扰的，受害人有权依法请求行为人 · 第195页
  [2] [2] 中华人民共和国宪法 · －1－ > 第四章 > 国旗、国歌、国徽、首都 > 一九一一年孙中山先生领导的辛亥革命，废除了封建帝制， · 第2页
  [3] [3] 中华人民共和国民法典 · 第四编 > 第二章 > 生命权、身体权和健康权 > 调查处置等措施，防止和制止利用职权、从属关系等实施性骚扰。 · 第195页
  [4] [4] 中华人民共和国宪法 · －1－ > 第四章 > 国旗、国歌、国徽、首都 > 在我国，剥削阶级作为阶级已经消灭，但是阶级斗争还将在 · 第4页

```

python evaluate.py --pipeline native
── 按题型统计 ──
  simple_fact               题数=4  拒绝率=25%  平均回答长度=113字
  precise_number            题数=6  拒绝率=33%  平均回答长度=70字
  cross_doc_compare         题数=5  拒绝率=40%  平均回答长度=245字
  time_trend                题数=3  拒绝率=67%  平均回答长度=146字
  should_refuse             题数=2  拒绝率=100%  平均回答长度=68字


python compare_strategies.py
2026-07-09 01:47:46,815 [INFO]     Hit@4=0.700  MRR=0.700

======================================================================
消融实验结果汇总（Top-4）
======================================================================
分块策略            检索方式              Hit Rate        MRR     题数
----------------------------------------------------------------------
semantic        vector_only          0.700      0.700     10
semantic        hybrid               0.700      0.700     10

2026-07-09 01:49:07,252 [INFO]     Hit@4=0.700  MRR=0.700


 python compare_strategies.py --strategies semantic,hierarchical
======================================================================
消融实验结果汇总（Top-4）
======================================================================
分块策略            检索方式              Hit Rate        MRR     题数
----------------------------------------------------------------------
semantic        vector_only          0.700      0.700     10
semantic        hybrid               0.700      0.700     10
hierarchical    vector_only          0.700      0.700     10
hierarchical    hybrid               0.700      0.700     10

python compare_strategies.py --modes vector_only,hybrid
======================================================================
消融实验结果汇总（Top-4）
======================================================================
分块策略            检索方式              Hit Rate        MRR     题数
----------------------------------------------------------------------
semantic        vector_only          0.700      0.700     10
semantic        hybrid               0.700      0.700     10