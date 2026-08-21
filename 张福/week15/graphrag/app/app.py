#!/usr/bin/env python3
"""Flask backend for 2024-2025 Energy Saving & Carbon Reduction Action Plan Knowledge Graph.

Provides REST API endpoints for querying the Neo4j knowledge graph and serves
the interactive HTML visualization page.

Neo4j connection: bolt://localhost:7687, user: neo4j
"""

import os
import time
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from flask import Flask, jsonify, request, render_template, Response, stream_with_context, send_from_directory

from neo4j import GraphDatabase

# GraphRAG: dashscope for LLM (Qwen via DashScope API)
try:
    import dashscope
    from dashscope import Generation

    # Read API key from environment, or fall back to .bashrc
    _api_key = os.environ.get("DASHSCOPE_API_KEY", "")
    if not _api_key:
        import re as _re
        _bashrc = os.path.expanduser("~/.bashrc")
        if os.path.exists(_bashrc):
            with open(_bashrc, "r") as _f:
                for _line in _f:
                    _m = _re.search(r'export\s+DASHSCOPE_API_KEY\s*=\s*"([^"]+)"', _line)
                    if _m:
                        _api_key = _m.group(1)
                        break
    dashscope.api_key = _api_key
    DASHSCOPE_AVAILABLE = bool(_api_key)
except ImportError:
    DASHSCOPE_AVAILABLE = False

# ============================================================================
# Configuration
# ============================================================================
NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "neo4j")

app = Flask(__name__, template_folder="templates", static_folder="static")
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


# ============================================================================
# Database helpers
# ============================================================================
def run_query(cypher, **params):
    """Execute a Cypher query and return records as list of dicts."""
    with driver.session() as session:
        result = session.run(cypher, **params)
        return [dict(r) for r in result]


def node_to_dict(node):
    """Convert a Neo4j node to a JSON-serializable dict."""
    labels = list(node.labels)
    return {
        "entity_id": node.get("entity_id", ""),
        "entity_name": node.get("entity_name", ""),
        "entity_type": node.get("entity_type", labels[0] if labels else ""),
        "labels": labels,
    }


def rel_to_dict(rel):
    """Convert a Neo4j relationship to a JSON-serializable dict."""
    return {
        "rel_id": rel.get("rel_id", ""),
        "rel_name": rel.get("rel_name", rel.type),
        "rel_desc": rel.get("rel_desc", ""),
        "rel_type": rel.type,
        "subject": rel.get("subject", ""),
        "object": rel.get("object", ""),
    }


# ============================================================================
# Routes — Pages
# ============================================================================
@app.route("/")
def index():
    """Serve the main HTML page."""
    return render_template("index.html")


# ============================================================================
# Routes — API
# ============================================================================
@app.route("/api/stats")
def api_stats():
    """Return overall graph statistics."""
    node_count = run_query("MATCH (n:Entity) RETURN count(n) AS c")[0]["c"]
    rel_count = run_query("MATCH ()-[r]->() RETURN count(r) AS c")[0]["c"]
    labels = run_query(
        "MATCH (n:Entity) "
        "UNWIND [l IN labels(n) WHERE l <> 'Entity'] AS lbl "
        "RETURN lbl AS label, count(n) AS cnt ORDER BY cnt DESC"
    )
    rel_types = run_query(
        "MATCH ()-[r]->() RETURN type(r) AS rtype, count(r) AS cnt ORDER BY cnt DESC"
    )
    return jsonify({
        "node_count": node_count,
        "rel_count": rel_count,
        "labels": labels,
        "rel_types": rel_types,
    })


@app.route("/api/graph")
def api_graph():
    """Return graph data (nodes + edges) for visualization.

    Optional query params:
      - limit: max nodes to return (default 100)
      - entity_type: filter by entity type label
    """
    limit = int(request.args.get("limit", 100))
    entity_type = request.args.get("entity_type", "")

    if entity_type:
        nodes_q = (
            "MATCH (n:Entity) WHERE n.entity_type = $etype "
            "RETURN n ORDER BY n.entity_id LIMIT $limit"
        )
        node_records = run_query(nodes_q, etype=entity_type, limit=limit)
    else:
        nodes_q = (
            "MATCH (n:Entity) RETURN n ORDER BY n.entity_id LIMIT $limit"
        )
        node_records = run_query(nodes_q, limit=limit)

    node_ids = [r["n"].get("entity_id") for r in node_records]
    if not node_ids:
        return jsonify({"nodes": [], "edges": []})

    # Get edges between the selected nodes
    edges_q = (
        "MATCH (a:Entity)-[r]->(b:Entity) "
        "WHERE a.entity_id IN $ids AND b.entity_id IN $ids "
        "RETURN a.entity_id AS source, r, b.entity_id AS target"
    )
    edge_records = run_query(edges_q, ids=node_ids)

    nodes = [node_to_dict(r["n"]) for r in node_records]
    edges = []
    for r in edge_records:
        rel = r["r"]
        edges.append({
            "source": r["source"],
            "target": r["target"],
            "rel_id": rel.get("rel_id", ""),
            "rel_name": rel.get("rel_name", rel.type),
            "rel_desc": rel.get("rel_desc", ""),
            "rel_type": rel.type,
        })

    return jsonify({"nodes": nodes, "edges": edges})


@app.route("/api/graph/around/<entity_id>")
def api_graph_around(entity_id):
    """Return subgraph around a specific entity (1-hop neighbors)."""
    nodes_q = (
        "MATCH (n:Entity {entity_id: $eid}) "
        "OPTIONAL MATCH (n)-[r]-(m:Entity) "
        "WITH collect(DISTINCT n) + collect(DISTINCT m) AS allNodes "
        "UNWIND allNodes AS node "
        "WITH DISTINCT node "
        "RETURN node"
    )
    node_records = run_query(nodes_q, eid=entity_id)

    valid_ids = set()
    nodes = []
    for r in node_records:
        if r["node"] is not None:
            nodes.append(node_to_dict(r["node"]))
            valid_ids.add(r["node"].get("entity_id"))

    edges_q = (
        "MATCH (a:Entity)-[r]->(b:Entity) "
        "WHERE a.entity_id IN $ids AND b.entity_id IN $ids "
        "RETURN a.entity_id AS source, r, b.entity_id AS target"
    )
    edge_records = run_query(edges_q, ids=list(valid_ids))

    edges = []
    for r in edge_records:
        rel = r["r"]
        edges.append({
            "source": r["source"],
            "target": r["target"],
            "rel_id": rel.get("rel_id", ""),
            "rel_name": rel.get("rel_name", rel.type),
            "rel_desc": rel.get("rel_desc", ""),
            "rel_type": rel.type,
        })

    return jsonify({"nodes": nodes, "edges": edges})


# --- Query 1: Search entity by name ---
@app.route("/api/query/entity")
def api_query_entity():
    """Query 1: Search entities by name.

    Params: name (entity name or partial match)
    Returns: matching entities + auto-generated Cypher
    """
    name = request.args.get("name", "").strip()
    if not name:
        return jsonify({
            "error": "请输入实体名称",
            "cypher": "// 请输入实体名称进行查询",
            "results": [],
        })

    cypher = (
        "MATCH (n:Entity) "
        "WHERE n.entity_name CONTAINS $name "
        "RETURN n.entity_id AS entity_id, n.entity_name AS entity_name, "
        "n.entity_type AS entity_type, labels(n) AS labels "
        "ORDER BY n.entity_name"
    )
    results = run_query(cypher, name=name)

    display_cypher = (
        "MATCH (n:Entity)\n"
        "WHERE n.entity_name CONTAINS '{name}'\n"
        "RETURN n.entity_id, n.entity_name, n.entity_type, labels(n)\n"
        "ORDER BY n.entity_name;"
    ).format(name=name)

    return jsonify({
        "cypher": display_cypher,
        "count": len(results),
        "results": results,
    })


# --- Query 2: Search relation by name ---
@app.route("/api/query/relation")
def api_query_relation():
    """Query 2: Search relationships by name.

    Params: name (relation name or partial match)
    Returns: matching relationships + auto-generated Cypher
    """
    name = request.args.get("name", "").strip()
    if not name:
        return jsonify({
            "error": "请输入关系名称",
            "cypher": "// 请输入关系名称进行查询",
            "results": [],
        })

    cypher = (
        "MATCH (a:Entity)-[r]->(b:Entity) "
        "WHERE r.rel_name CONTAINS $name OR type(r) CONTAINS $name "
        "RETURN a.entity_id AS sub_id, a.entity_name AS subject, "
        "r.rel_id AS rel_id, r.rel_name AS rel_name, r.rel_desc AS rel_desc, "
        "type(r) AS rel_type, "
        "b.entity_id AS obj_id, b.entity_name AS object "
        "ORDER BY r.rel_name, a.entity_name"
    )
    results = run_query(cypher, name=name)

    display_cypher = (
        "MATCH (a:Entity)-[r]->(b:Entity)\n"
        "WHERE r.rel_name CONTAINS '{name}' OR type(r) CONTAINS '{name}'\n"
        "RETURN a.entity_name AS subject, r.rel_name AS relation, "
        "b.entity_name AS object, r.rel_desc AS description\n"
        "ORDER BY r.rel_name;"
    ).format(name=name)

    return jsonify({
        "cypher": display_cypher,
        "count": len(results),
        "results": results,
    })


# --- Query 3: Search triple ---
@app.route("/api/query/triple")
def api_query_triple():
    """Query 3: Search triples by subject / relation / object.

    Params: subject, relation, object (all optional, at least one required)
    Returns: matching triples + auto-generated Cypher
    """
    subject = request.args.get("subject", "").strip()
    relation = request.args.get("relation", "").strip()
    obj = request.args.get("object", "").strip()

    if not any([subject, relation, obj]):
        return jsonify({
            "error": "请至少输入主体、关系或客体之一",
            "cypher": "// 请至少输入一个查询条件",
            "results": [],
        })

    conditions = []
    params = {}
    if subject:
        conditions.append("a.entity_name CONTAINS $subject")
        params["subject"] = subject
    if relation:
        conditions.append("(r.rel_name CONTAINS $relation OR type(r) CONTAINS $relation)")
        params["relation"] = relation
    if obj:
        conditions.append("b.entity_name CONTAINS $object")
        params["object"] = obj

    where_clause = " AND ".join(conditions)

    cypher = (
        "MATCH (a:Entity)-[r]->(b:Entity) "
        "WHERE {where} "
        "RETURN a.entity_id AS sub_id, a.entity_name AS subject, "
        "a.entity_type AS sub_type, "
        "r.rel_id AS rel_id, r.rel_name AS rel_name, r.rel_desc AS rel_desc, "
        "type(r) AS rel_type, "
        "b.entity_id AS obj_id, b.entity_name AS object, "
        "b.entity_type AS obj_type "
        "ORDER BY a.entity_name, r.rel_name"
    ).format(where=where_clause)

    results = run_query(cypher, **params)

    # Build display cypher with literal values
    display_conditions = []
    if subject:
        display_conditions.append("a.entity_name CONTAINS '{}'".format(subject))
    if relation:
        display_conditions.append(
            "(r.rel_name CONTAINS '{}' OR type(r) CONTAINS '{}')".format(relation, relation)
        )
    if obj:
        display_conditions.append("b.entity_name CONTAINS '{}'".format(obj))
    display_where = " AND ".join(display_conditions)

    display_cypher = (
        "MATCH (a:Entity)-[r]->(b:Entity)\n"
        "WHERE {where}\n"
        "RETURN a.entity_name AS subject, r.rel_name AS relation, "
        "b.entity_name AS object, r.rel_desc AS description\n"
        "ORDER BY a.entity_name, r.rel_name;"
    ).format(where=display_where)

    return jsonify({
        "cypher": display_cypher,
        "count": len(results),
        "results": results,
    })


# --- Entity detail ---
@app.route("/api/entity/<entity_id>")
def api_entity_detail(entity_id):
    """Get entity details + its relationships (incoming & outgoing)."""
    node_q = "MATCH (n:Entity {entity_id: $eid}) RETURN n"
    node_records = run_query(node_q, eid=entity_id)
    if not node_records:
        return jsonify({"error": "实体不存在"}), 404

    node = node_to_dict(node_records[0]["n"])

    # Outgoing relationships
    out_q = (
        "MATCH (n:Entity {entity_id: $eid})-[r]->(m:Entity) "
        "RETURN r.rel_id AS rel_id, r.rel_name AS rel_name, r.rel_desc AS rel_desc, "
        "type(r) AS rel_type, m.entity_id AS obj_id, m.entity_name AS obj_name, "
        "m.entity_type AS obj_type"
    )
    outgoing = run_query(out_q, eid=entity_id)

    # Incoming relationships
    in_q = (
        "MATCH (m:Entity)-[r]->(n:Entity {entity_id: $eid}) "
        "RETURN r.rel_id AS rel_id, r.rel_name AS rel_name, r.rel_desc AS rel_desc, "
        "type(r) AS rel_type, m.entity_id AS sub_id, m.entity_name AS sub_name, "
        "m.entity_type AS sub_type"
    )
    incoming = run_query(in_q, eid=entity_id)

    return jsonify({
        "entity": node,
        "outgoing": outgoing,
        "incoming": incoming,
    })


# --- All entity types (for filter dropdown) ---
@app.route("/api/entity_types")
def api_entity_types():
    """Return all entity type labels for filtering."""
    records = run_query(
        "MATCH (n:Entity) "
        "UNWIND [l IN labels(n) WHERE l <> 'Entity'] AS lbl "
        "RETURN DISTINCT lbl AS type ORDER BY lbl"
    )
    return jsonify({"types": [r["type"] for r in records]})


# --- All relations ---
@app.route("/api/relations")
def api_relations():
    """Return all distinct relation types."""
    records = run_query(
        "MATCH ()-[r]->() "
        "RETURN DISTINCT r.rel_name AS name, r.rel_desc AS desc, type(r) AS rtype "
        "ORDER BY r.rel_name"
    )
    return jsonify({"relations": records})


# ============================================================================
# GraphRAG: Two distinct query methods
#   1. Local Search  — entity-centric, retrieves 1-hop neighborhood
#   2. Global Search — community-summary-vector-based map-reduce retrieval
# ============================================================================
import re as _re
import json as _json
import numpy as np
import networkx as nx
from dashscope import TextEmbedding

EMBEDDING_MODEL = "text-embedding-v3"
COMMUNITIES_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "communities.json"
)

LLM_SYSTEM_PROMPT = (
    "你是一个节能降碳政策知识助手，基于知识图谱中的信息回答用户问题。"
    "请仅根据提供的知识图谱上下文信息回答问题，不要编造信息。"
    "如果上下文中没有相关信息，请说明无法从知识图谱中找到答案。"
    "回答时请引用具体的实体名称和关系。"
)


# ---- Shared utilities --------------------------------------------------

def extract_keywords(question):
    """Extract keywords from user question by matching against known entity names."""
    entities = run_query(
        "MATCH (n:Entity) RETURN n.entity_id AS id, n.entity_name AS name, "
        "n.entity_type AS type ORDER BY size(n.entity_name) DESC"
    )
    matched = []
    for e in entities:
        if e["name"] and e["name"] in question:
            matched.append({"entity_id": e["id"], "name": e["name"], "type": e["type"]})

    chinese_chars = _re.findall(r'[\u4e00-\u9fff]', question)
    general = []
    for length in (2, 3):
        for i in range(len(chinese_chars) - length + 1):
            general.append(''.join(chinese_chars[i:i + length]))
    _seen = set()
    general = [x for x in general if not (x in _seen or _seen.add(x))]
    stop_words = {"什么", "怎么", "如何", "哪些", "那个", "哪个", "请问", "可以",
                  "已经", "通过", "进行", "实施", "关于", "根据", "按照", "以及",
                  "有哪", "哪些", "些节", "降碳", "目标", "行业", "行动", "多少",
                  "是非", "非化", "化石", "石能", "能源", "消费", "占比", "占比"}
    general = [w for w in general if w not in stop_words]
    return {"matched_entities": matched, "general_keywords": general}


def call_llm(question, context, system_prompt=None):
    """Call Qwen LLM via DashScope API."""
    if not DASHSCOPE_AVAILABLE:
        return "错误: DASHSCOPE_API_KEY 未设置或 dashscope SDK 未安装。"
    sys_p = system_prompt or LLM_SYSTEM_PROMPT
    user_p = (
        f"知识图谱上下文信息：\n{context}\n\n"
        f"用户问题：{question}\n\n"
        "请基于上述知识图谱信息回答用户问题："
    )
    try:
        resp = Generation.call(
            model="qwen-plus",
            messages=[
                {"role": "system", "content": sys_p},
                {"role": "user", "content": user_p},
            ],
            result_format="message",
        )
        if resp.status_code == 200:
            return resp.output.choices[0].message.content
        return f"LLM 调用失败: {resp.code} - {resp.message}"
    except Exception as e:
        return f"LLM 调用异常: {str(e)}"


def get_embedding(text):
    """Get text embedding vector via DashScope TextEmbedding API."""
    if not DASHSCOPE_AVAILABLE:
        return None
    try:
        resp = TextEmbedding.call(model=EMBEDDING_MODEL, input=text)
        if resp.status_code == 200:
            return resp.output["embeddings"][0]["embedding"]
        return None
    except Exception:
        return None


def cosine_sim(v1, v2):
    """Cosine similarity between two lists."""
    a, b = np.array(v1), np.array(v2)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


# ---- Community building (for Global Search) ----------------------------

def build_communities():
    """Build community summaries with vector embeddings.

    Pipeline:
      1. Load graph from Neo4j → networkx
      2. Run Louvain community detection
      3. For each community: collect entities+relationships → LLM summary
      4. Embed each summary via DashScope TextEmbedding
      5. Persist to data/communities.json
    Returns a dict with build statistics.
    """
    # 1. Load graph into networkx
    nodes = run_query("MATCH (n:Entity) RETURN n.entity_id AS id, n.entity_name AS name, n.entity_type AS type")
    edges = run_query(
        "MATCH (a:Entity)-[r]->(b:Entity) "
        "RETURN a.entity_id AS src, b.entity_id AS tgt, "
        "r.rel_name AS rel, r.rel_desc AS desc"
    )

    G = nx.Graph()
    for n in nodes:
        G.add_node(n["id"], name=n["name"], type=n["type"])
    for e in edges:
        G.add_edge(e["src"], e["tgt"], rel=e["rel"], desc=e["desc"])

    # 2. Louvain community detection
    communities_raw = nx.community.louvain_communities(G, seed=42)
    communities_list = [list(c) for c in communities_raw]

    # 3+4. For each community: generate summary + embedding
    community_data = []
    for idx, member_ids in enumerate(communities_list):
        member_nodes = [G.nodes[mid] for mid in member_ids if mid in G.nodes]
        entity_names = [mn["name"] for mn in member_nodes if mn.get("name")]
        entity_types = sorted(set(mn.get("type", "") for mn in member_nodes))

        # Collect internal relationships
        rels = []
        for u, v in G.subgraph(member_ids).edges():
            ed = G.edges[u, v]
            rels.append(f"{G.nodes[u].get('name','?')} —{ed.get('rel','?')}→ {G.nodes[v].get('name','?')}")

        # Build text for LLM summary
        rel_text = "\n".join(rels[:40]) if rels else "（无内部关系）"
        entity_text = ", ".join(entity_names[:30])
        summary_input = (
            f"社区 #{idx}\n"
            f"实体类型: {', '.join(entity_types)}\n"
            f"实体: {entity_text}\n"
            f"关系:\n{rel_text}\n\n"
            "请用2-3句话概括该社区的核心主题和关键信息。"
        )

        summary = call_llm(
            "请概括以下知识图谱社区的核心主题。",
            summary_input,
            system_prompt="你是一个知识图谱分析助手。请简洁地概括社区主题。",
        )

        # Embed the summary
        embedding = get_embedding(summary)

        community_data.append({
            "community_id": idx,
            "entity_count": len(member_ids),
            "entity_names": entity_names,
            "entity_types": entity_types,
            "relationships": rels,
            "summary": summary,
            "embedding": embedding,
        })

    # 5. Persist
    os.makedirs(os.path.dirname(COMMUNITIES_FILE), exist_ok=True)
    output = {
        "built_at": _re.sub(r'\.\d+$', '', str(__import__("datetime").datetime.now())),
        "algorithm": "louvain",
        "embedding_model": EMBEDDING_MODEL,
        "total_entities": len(nodes),
        "total_edges": len(edges),
        "community_count": len(communities_list),
        "communities": community_data,
    }
    with open(COMMUNITIES_FILE, "w", encoding="utf-8") as f:
        _json.dump(output, f, ensure_ascii=False, indent=2)

    return output


def load_communities():
    """Load community data from file. Returns None if not built."""
    if not os.path.exists(COMMUNITIES_FILE):
        return None
    with open(COMMUNITIES_FILE, "r", encoding="utf-8") as f:
        return _json.load(f)


# ---- Method 1: Local Search (entity-centric) --------------------------

def graphrag_local_search(keywords):
    """Local Search: find matching entities → retrieve 1-hop neighborhood → context."""
    matched = keywords["matched_entities"]
    general_kw = keywords["general_keywords"]
    search_terms = [e["name"] for e in matched] + general_kw
    if not search_terms:
        return {"cypher": "// 无可搜索的关键词", "params": {}, "results": [], "context": "未找到相关实体。"}

    conditions = []
    params = {}
    for i, term in enumerate(search_terms[:10]):
        pname = f"term{i}"
        conditions.append(f"n.entity_name CONTAINS ${pname}")
        params[pname] = term
    where_clause = " OR ".join(conditions)

    cypher = (
        f"MATCH (n:Entity) WHERE {where_clause} "
        "OPTIONAL MATCH (n)-[r_out]->(m_out:Entity) "
        "OPTIONAL MATCH (m_in:Entity)-[r_in]->(n) "
        "RETURN n.entity_id AS entity_id, n.entity_name AS entity_name, "
        "n.entity_type AS entity_type, "
        "r_out.rel_name AS out_rel, m_out.entity_name AS out_target, "
        "m_out.entity_type AS out_target_type, "
        "r_in.rel_name AS in_rel, m_in.entity_name AS in_source, "
        "m_in.entity_type AS in_source_type "
        "ORDER BY n.entity_name"
    )
    rows = run_query(cypher, **params)

    entity_map = {}
    for row in rows:
        eid = row["entity_id"]
        if eid not in entity_map:
            entity_map[eid] = {
                "entity_id": eid, "entity_name": row["entity_name"],
                "entity_type": row["entity_type"], "out_rels": [], "in_rels": [],
            }
        if row["out_rel"]:
            entity_map[eid]["out_rels"].append(
                {"rel": row["out_rel"], "target": row["out_target"], "target_type": row["out_target_type"]})
        if row["in_rel"]:
            entity_map[eid]["in_rels"].append(
                {"rel": row["in_rel"], "source": row["in_source"], "source_type": row["in_source_type"]})

    results = list(entity_map.values())
    ctx_lines = []
    for r in results:
        ctx_lines.append(f"【{r['entity_type']}】{r['entity_name']} (ID: {r['entity_id']})")
        for rel in r["out_rels"]:
            ctx_lines.append(f"  → {rel['rel']} → {rel['target']} [{rel['target_type']}]")
        for rel in r["in_rels"]:
            ctx_lines.append(f"  ← {rel['rel']} ← {rel['source']} [{rel['source_type']}]")
    context = "\n".join(ctx_lines) if ctx_lines else "局部搜索未找到相关实体关系。"

    return {"cypher": cypher, "params": params, "results": results, "context": context}


# ---- Method 2: Global Search (community-summary-vector map-reduce) ----

def graphrag_global_search(question, top_k=5):
    """Global Search: embed question → match community summaries → map-reduce answer.

    Returns a dict with each step's data for UI display.
    """
    data = load_communities()
    if not data:
        return {"error": "社区摘要尚未构建，请先点击「构建社区摘要」按钮。"}

    communities = data["communities"]

    # Filter communities with valid embeddings
    valid = [c for c in communities if c.get("embedding")]
    if not valid:
        return {"error": "社区摘要向量数据无效，请重新构建。"}

    # Step A: Embed the question
    q_embedding = get_embedding(question)
    if not q_embedding:
        return {"error": "问题向量化失败，请检查 DashScope API。"}

    # Step B: Cosine similarity with each community summary
    scored = []
    for c in valid:
        score = cosine_sim(q_embedding, c["embedding"])
        scored.append((c["community_id"], score, c))
    scored.sort(key=lambda x: x[1], reverse=True)

    # Step C: Select top-K communities
    top_communities = scored[:top_k]

    # Step D (Map): For each selected community, generate intermediate answer
    map_results = []
    for cid, score, comm in top_communities:
        summary = comm["summary"]
        intermediate = call_llm(
            question,
            f"社区摘要：\n{summary}\n\n社区实体：{', '.join(comm['entity_names'][:20])}",
            system_prompt=(
                "你是一个知识图谱分析助手。根据以下社区摘要信息，"
                "简要回答用户问题。如果该社区信息与问题无关，回复「该社区无相关信息」。"
            ),
        )
        map_results.append({
            "community_id": cid,
            "similarity": round(score, 4),
            "summary": summary,
            "entity_count": comm["entity_count"],
            "intermediate_answer": intermediate,
        })

    # Step E (Reduce): Aggregate intermediate answers
    reduce_context = "\n\n".join(
        f"--- 社区 #{mr['community_id']} (相似度: {mr['similarity']}) ---\n"
        f"摘要: {mr['summary']}\n"
        f"中间回答: {mr['intermediate_answer']}"
        for mr in map_results
    )
    final_answer = call_llm(
        question,
        f"以下是多个知识图谱社区的摘要及其中间回答：\n{reduce_context}",
        system_prompt=(
            "你是一个节能降碳政策知识助手。请综合以下多个社区的信息，"
            "为用户问题生成一个完整、准确的最终回答。"
            "整合不同社区的视角，去除重复信息，保持逻辑清晰。"
        ),
    )

    return {
        "question_embedding_dim": len(q_embedding),
        "total_communities": len(valid),
        "top_k": top_k,
        "scored_communities": [
            {"community_id": cid, "similarity": round(s, 4),
             "summary": c["summary"][:100], "entity_count": c["entity_count"]}
            for cid, s, c in scored
        ],
        "selected_communities": [
            {"community_id": mr["community_id"], "similarity": mr["similarity"],
             "summary": mr["summary"], "entity_count": mr["entity_count"],
             "intermediate_answer": mr["intermediate_answer"]}
            for mr in map_results
        ],
        "reduce_context": reduce_context,
        "final_answer": final_answer,
    }


# ---- API endpoints ----------------------------------------------------

@app.route("/api/graphrag/community-status")
def api_community_status():
    """Check if community summaries have been built."""
    data = load_communities()
    if not data:
        return jsonify({"built": False})
    return jsonify({
        "built": True,
        "built_at": data.get("built_at", ""),
        "algorithm": data.get("algorithm", ""),
        "embedding_model": data.get("embedding_model", ""),
        "community_count": data.get("community_count", 0),
        "total_entities": data.get("total_entities", 0),
        "total_edges": data.get("total_edges", 0),
    })


@app.route("/api/graphrag/build-communities", methods=["POST"])
def api_build_communities():
    """Build community summaries with vector embeddings."""
    try:
        result = build_communities()
        return jsonify({
            "success": True,
            "built_at": result["built_at"],
            "algorithm": result["algorithm"],
            "embedding_model": result["embedding_model"],
            "community_count": result["community_count"],
            "total_entities": result["total_entities"],
            "total_edges": result["total_edges"],
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/graphrag/local-search", methods=["POST"])
def api_local_search():
    """Method 1 — Local Search: entity-centric retrieval + LLM answer."""
    data = request.get_json()
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "请输入问题"})

    steps = []

    # Step 1: Keyword extraction
    keywords = extract_keywords(question)
    steps.append({
        "step": 1, "title": "步骤1：关键词提取",
        "description": "从用户问题中提取关键词，匹配知识图谱中的实体名称",
        "input": {"question": question},
        "output": {
            "matched_entities": keywords["matched_entities"],
            "general_keywords": keywords["general_keywords"],
        },
    })

    # Step 2: Entity matching + 1-hop neighborhood retrieval
    local_result = graphrag_local_search(keywords)
    steps.append({
        "step": 2, "title": "步骤2：实体匹配 + 1跳邻域检索",
        "description": "通过 Cypher 在 Neo4j 中查找匹配实体，获取其出度/入度关系",
        "sent": {"cypher": local_result["cypher"], "params": local_result.get("params", {})},
        "returned": {"entity_count": len(local_result["results"]), "results": local_result["results"]},
        "assembled_context": local_result["context"],
    })

    # Step 3: Assemble context
    context = local_result["context"]
    steps.append({
        "step": 3, "title": "步骤3：组装 Context",
        "description": "将实体及其邻域关系组装为上下文信息",
        "assembled_context": context,
    })

    # Step 4: LLM answer
    user_prompt = (
        f"知识图谱上下文信息：\n{context}\n\n"
        f"用户问题：{question}\n\n请基于上述知识图谱信息回答用户问题："
    )
    answer = call_llm(question, context)
    steps.append({
        "step": 4, "title": "步骤4：调用 LLM 生成回答",
        "description": "通过 DASHSCOPE_API_KEY 调用通义千问(Qwen-plus)模型，基于 Context 生成回答",
        "sent": {"model": "qwen-plus", "system_prompt": LLM_SYSTEM_PROMPT, "user_prompt": user_prompt},
        "returned": {"answer": answer},
    })

    return jsonify({
        "method": "local", "question": question, "steps": steps,
        "answer": answer, "dashscope_available": DASHSCOPE_AVAILABLE,
    })


@app.route("/api/graphrag/global-search", methods=["POST"])
def api_global_search():
    """Method 2 — Global Search: community-summary-vector map-reduce."""
    data = request.get_json()
    question = data.get("question", "").strip()
    top_k = data.get("top_k", 5)
    if not question:
        return jsonify({"error": "请输入问题"})

    steps = []

    # Step 1: Embed the question
    q_embedding = get_embedding(question)
    steps.append({
        "step": 1, "title": "步骤1：问题向量化",
        "description": f"通过 DashScope TextEmbedding ({EMBEDDING_MODEL}) 将用户问题转换为向量",
        "sent": {"model": EMBEDDING_MODEL, "text": question},
        "returned": {"embedding_dim": len(q_embedding) if q_embedding else 0, "success": bool(q_embedding)},
    })

    if not q_embedding:
        return jsonify({"error": "问题向量化失败，请检查 DashScope API。", "steps": steps})

    # Step 2: Vector similarity matching with community summaries
    comm_data = load_communities()
    if not comm_data:
        return jsonify({"error": "社区摘要尚未构建，请先构建社区摘要。", "steps": steps})

    communities = [c for c in comm_data["communities"] if c.get("embedding")]
    scored = []
    for c in communities:
        score = cosine_sim(q_embedding, c["embedding"])
        scored.append({"community_id": c["community_id"], "similarity": round(score, 4),
                       "summary_preview": c["summary"][:100], "entity_count": c["entity_count"]})
    scored.sort(key=lambda x: x["similarity"], reverse=True)

    steps.append({
        "step": 2, "title": "步骤2：社区摘要向量匹配",
        "description": f"计算问题向量与 {len(communities)} 个社区摘要向量的余弦相似度",
        "sent": {"metric": "cosine_similarity", "total_communities": len(communities)},
        "returned": {"all_scores": scored},
    })

    # Step 3: Select top-K communities
    top_k_actual = min(top_k, len(scored))
    selected = scored[:top_k_actual]
    steps.append({
        "step": 3, "title": f"步骤3：选择 Top-{top_k_actual} 社区",
        "description": "选取相似度最高的社区摘要作为上下文",
        "selected": selected,
    })

    # Step 4 (Map): Per-community intermediate answers
    full_communities = {c["community_id"]: c for c in communities}
    map_results = []
    for s in selected:
        c = full_communities[s["community_id"]]
        intermediate = call_llm(
            question,
            f"社区摘要：\n{c['summary']}\n\n社区实体：{', '.join(c['entity_names'][:20])}",
            system_prompt=(
                "你是一个知识图谱分析助手。根据以下社区摘要信息，"
                "简要回答用户问题。如果该社区信息与问题无关，回复「该社区无相关信息」。"
            ),
        )
        map_results.append({
            "community_id": s["community_id"], "similarity": s["similarity"],
            "summary": c["summary"], "entity_count": c["entity_count"],
            "intermediate_answer": intermediate,
        })
    steps.append({
        "step": 4, "title": f"步骤4 (Map)：逐社区生成中间回答 ({len(map_results)} 个)",
        "description": "对每个选中的社区，用其摘要生成针对用户问题的中间回答",
        "sent": {"model": "qwen-plus", "count": len(map_results)},
        "returned": {"map_results": map_results},
    })

    # Step 5 (Reduce): Aggregate
    reduce_context = "\n\n".join(
        f"--- 社区 #{mr['community_id']} (相似度: {mr['similarity']}) ---\n"
        f"摘要: {mr['summary']}\n中间回答: {mr['intermediate_answer']}"
        for mr in map_results
    )
    final_answer = call_llm(
        question,
        f"以下是多个知识图谱社区的摘要及其中间回答：\n{reduce_context}",
        system_prompt=(
            "你是一个节能降碳政策知识助手。请综合以下多个社区的信息，"
            "为用户问题生成一个完整、准确的最终回答。"
            "整合不同社区的视角，去除重复信息，保持逻辑清晰。"
        ),
    )
    steps.append({
        "step": 5, "title": "步骤5 (Reduce)：汇总生成最终回答",
        "description": "将所有社区的中间回答汇总，通过 LLM 生成最终回答",
        "sent": {"model": "qwen-plus", "reduce_context": reduce_context},
        "returned": {"answer": final_answer},
    })

    return jsonify({
        "method": "global", "question": question, "steps": steps,
        "answer": final_answer, "dashscope_available": DASHSCOPE_AVAILABLE,
    })


# ============================================================================
# Subagent Parallel Query System
#   Main Agent dispatches Local Search + Global Search subagents in parallel,
#   then merges results and calls LLM for final answer.
#   Uses SSE (Server-Sent Events) for real-time status streaming.
# ============================================================================

SUBAGENT_TEMPLATE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "subagent", "templates"
)


@app.route("/subagent")
def subagent_page():
    """Serve the subagent parallel query HTML page."""
    return send_from_directory(SUBAGENT_TEMPLATE_DIR, "subagent.html")


def _ts_dict():
    """Return timestamp dict with epoch and human-readable string."""
    t = time.time()
    return {
        "epoch": round(t, 3),
        "str": time.strftime("%H:%M:%S", time.localtime(t))
        + f".{int(t * 1000) % 1000:03d}",
    }


@app.route("/api/subagent/parallel-query", methods=["POST"])
def api_subagent_parallel_query():
    """Subagent parallel query endpoint using SSE for real-time updates.

    Pipeline:
      1. Main Agent receives question → dispatches two subagents in parallel
      2. Subagent A (Local Search): keyword extraction → entity neighborhood → LLM
      3. Subagent B (Global Search): question embedding → community vector match → Map-Reduce
      4. Main Agent merges both results → LLM generates final integrated answer

    Returns: text/event-stream with JSON events for each step.
    """
    data = request.get_json()
    question = data.get("question", "").strip()
    top_k = int(data.get("top_k", 5))

    if not question:
        return jsonify({"error": "请输入问题"}), 400

    def generate():
        event_queue = queue.Queue()
        done_flag = threading.Event()

        def emit(event_type, payload):
            event_queue.put({"event": event_type, "data": payload, "ts": _ts_dict()})

        def worker():
            results = {}

            # ---- Subagent A: Local Search ----
            def local_agent():
                start = time.time()
                emit("agent_update", {
                    "agent_id": "local",
                    "agent_name": "Local Search 子代理",
                    "status": "running",
                    "step": "关键词提取 + 实体邻域检索",
                })

                try:
                    keywords = extract_keywords(question)
                    emit("agent_step", {
                        "agent_id": "local",
                        "step": "关键词提取完成",
                        "detail": {
                            "matched_entities": len(keywords["matched_entities"]),
                            "general_keywords": len(keywords["general_keywords"]),
                        },
                    })

                    local_result = graphrag_local_search(keywords)
                    emit("agent_step", {
                        "agent_id": "local",
                        "step": "Cypher 1跳邻域检索完成",
                        "detail": {"entity_count": len(local_result["results"])},
                    })

                    answer = call_llm(question, local_result["context"])
                    elapsed = round(time.time() - start, 2)

                    results["local"] = {
                        "answer": answer,
                        "context": local_result["context"],
                        "elapsed": elapsed,
                        "entity_count": len(local_result["results"]),
                    }

                    emit("agent_update", {
                        "agent_id": "local",
                        "agent_name": "Local Search 子代理",
                        "status": "completed",
                        "elapsed": elapsed,
                        "result": answer,
                    })
                except Exception as e:
                    emit("agent_update", {
                        "agent_id": "local",
                        "agent_name": "Local Search 子代理",
                        "status": "error",
                        "error": str(e),
                    })

            # ---- Subagent B: Global Search ----
            def global_agent():
                start = time.time()
                emit("agent_update", {
                    "agent_id": "global",
                    "agent_name": "Global Search 子代理",
                    "status": "running",
                    "step": "社区摘要向量匹配 + Map-Reduce",
                })

                try:
                    gs_result = graphrag_global_search(question, top_k)
                    if "error" in gs_result:
                        emit("agent_update", {
                            "agent_id": "global",
                            "agent_name": "Global Search 子代理",
                            "status": "error",
                            "error": gs_result["error"],
                        })
                        return

                    emit("agent_step", {
                        "agent_id": "global",
                        "step": "Map-Reduce 完成",
                        "detail": {
                            "total_communities": gs_result.get("total_communities", 0),
                            "top_k": gs_result.get("top_k", 0),
                        },
                    })

                    elapsed = round(time.time() - start, 2)
                    results["global"] = {
                        "answer": gs_result["final_answer"],
                        "elapsed": elapsed,
                        "selected_communities": gs_result.get("selected_communities", []),
                    }

                    emit("agent_update", {
                        "agent_id": "global",
                        "agent_name": "Global Search 子代理",
                        "status": "completed",
                        "elapsed": elapsed,
                        "result": gs_result["final_answer"],
                    })
                except Exception as e:
                    emit("agent_update", {
                        "agent_id": "global",
                        "agent_name": "Global Search 子代理",
                        "status": "error",
                        "error": str(e),
                    })

            # ---- Main Agent: dispatch + merge ----
            main_start = time.time()
            emit("agent_update", {
                "agent_id": "main",
                "agent_name": "主代理 (Main Agent)",
                "status": "running",
                "step": "派发并行查询任务给子代理",
            })

            # Run both subagents in parallel
            with ThreadPoolExecutor(max_workers=2) as executor:
                fut_local = executor.submit(local_agent)
                fut_global = executor.submit(global_agent)
                fut_local.result()
                fut_global.result()

            # Main Agent merges results
            emit("agent_update", {
                "agent_id": "main",
                "agent_name": "主代理 (Main Agent)",
                "status": "merging",
                "step": "合并子代理结果 + 调用 LLM 生成最终回答",
            })

            parallel_elapsed = round(time.time() - main_start, 2)
            local_elapsed = results.get("local", {}).get("elapsed", 0)
            global_elapsed = results.get("global", {}).get("elapsed", 0)

            merge_start = time.time()
            if "local" in results and "global" in results:
                merge_context = (
                    "=== Local Search 子代理检索结果 ===\n"
                    f"{results['local']['answer']}\n\n"
                    "=== Global Search 子代理检索结果 ===\n"
                    f"{results['global']['answer']}"
                )
                final_answer = call_llm(
                    question,
                    merge_context,
                    system_prompt=(
                        "你是一个节能降碳政策知识助手。以下是两个子代理"
                        "（Local Search 基于实体邻域检索、Global Search 基于社区摘要向量检索）"
                        "分别基于知识图谱检索得到的回答。"
                        "请综合两个子代理的回答，去除重复信息，整合互补信息，"
                        "生成一个完整、准确、有逻辑的最终回答。"
                        "回答时请引用具体的实体名称和关系。"
                    ),
                )
            elif "local" in results:
                merge_context = results["local"]["answer"]
                final_answer = results["local"]["answer"]
            elif "global" in results:
                merge_context = results["global"]["answer"]
                final_answer = results["global"]["answer"]
            else:
                merge_context = ""
                final_answer = "两个子代理均未返回有效结果。"

            merge_elapsed = round(time.time() - merge_start, 2)
            total_elapsed = round(time.time() - main_start, 2)

            merge_data = {
                "answer": final_answer,
                "local_answer": results.get("local", {}).get("answer"),
                "global_answer": results.get("global", {}).get("answer"),
                "local_elapsed": local_elapsed,
                "global_elapsed": global_elapsed,
                "parallel_elapsed": parallel_elapsed,
                "merge_elapsed": merge_elapsed,
                "total_elapsed": total_elapsed,
            }
            if "local" not in results and "global" in results:
                merge_data["note"] = "Local Search 子代理未返回结果，仅使用 Global Search 结果"
            elif "global" not in results and "local" in results:
                merge_data["note"] = "Global Search 子代理未返回结果，仅使用 Local Search 结果"
            elif "local" not in results and "global" not in results:
                merge_data["error"] = True

            emit("merge_complete", merge_data)

            emit("agent_update", {
                "agent_id": "main",
                "agent_name": "主代理 (Main Agent)",
                "status": "completed",
                "elapsed": total_elapsed,
            })
            done_flag.set()

        # Start worker thread
        threading.Thread(target=worker, daemon=True).start()

        # Stream events via SSE
        while not done_flag.is_set() or not event_queue.empty():
            try:
                evt = event_queue.get(timeout=0.5)
                yield f"data: {_json.dumps(evt, ensure_ascii=False)}\n\n"
            except queue.Empty:
                yield ": keepalive\n\n"

        # Final done event
        yield f"data: {_json.dumps({'event': 'done', 'data': {}, 'ts': _ts_dict()}, ensure_ascii=False)}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


if __name__ == "__main__":
    # Verify connection on startup
    try:
        stats = run_query("MATCH (n:Entity) RETURN count(n) AS c")[0]
        print(f"Connected to Neo4j. Nodes: {stats['c']}")
    except Exception as e:
        print(f"Neo4j connection error: {e}")

    app.run(host="0.0.0.0", port=5000, debug=True)
