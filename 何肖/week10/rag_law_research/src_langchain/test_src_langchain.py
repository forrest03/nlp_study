import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "rag_chain_lc",
    Path("D:/ck/project_info/rag_annual_report/src_langchain/rag_chain_lc.py")
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

embeddings  = module.get_embeddings()
vectorstore = module.get_vectorstore(embeddings)
chain, retriever = module.build_chain(vectorstore)

# 问答
answer = chain.invoke("“公民有哪些基本权利？”")

# 单独检索（获取 contexts）
docs = retriever.invoke("“公民有哪些基本权利？”")
contexts = [doc.page_content for doc in docs]

print(answer)
print(contexts)