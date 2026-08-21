import requests

resp = requests.post(
    "http://localhost:8000/query",
    json={"question": "我被前公司领导性骚扰，已经离职了还能告他吗？"},
)
data = resp.json()
print(data["answer"])
for c in data["citations"]:
    print(f"  [{c['index']}] {c['source']}")