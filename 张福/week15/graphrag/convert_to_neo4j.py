#!/usr/bin/env python3
"""Convert knowledge graph JSON to Neo4j database file (.neo4jdb) in Cypher format.

Input:  graphrag/resources/graphs/2024-2025年节能降碳行动方案知识图谱.json
Output: graphrag/resources/graphs_neo4j/2024-2025-节能降碳行动方案.neo4jdb

The .neo4jdb file contains Cypher CREATE/MERGE statements that can be loaded
via `cypher-shell -f file.neo4jdb` or pasted into Neo4j Browser to reconstruct
the full knowledge graph database (nodes with labels + typed relationships).
"""

import json
import os
import re

# ============================================================================
# Configuration
# ============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(
    SCRIPT_DIR, "resources", "graphs", "2024-2025年节能降碳行动方案知识图谱.json"
)
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "resources", "graphs_neo4j")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "2024-2025-节能降碳行动方案.neo4jdb")


# ============================================================================
# Helpers
# ============================================================================
def escape_cypher_string(value: str) -> str:
    """Escape a string for safe embedding in Cypher single-quoted literals."""
    if value is None:
        return ""
    # Backslash first, then single quote, then other control chars
    value = str(value)
    value = value.replace("\\", "\\\\")
    value = value.replace("'", "\\'")
    value = value.replace("\n", "\\n")
    value = value.replace("\r", "\\r")
    value = value.replace("\t", "\\t")
    return value


def sanitize_label(label: str) -> str:
    """Sanitize an entity type / relation name into a valid Neo4j label.

    Neo4j labels/types use CamelCase; keep CJK characters (valid in Neo4j 4+)
    but strip punctuation that would break the syntax.
    """
    if not label:
        return "Entity"
    # Remove characters that are problematic in labels/types without backticks
    cleaned = re.sub(r"[`{}\[\]():,;\.]", "", label)
    # Collapse whitespace
    cleaned = re.sub(r"\s+", "", cleaned)
    return cleaned if cleaned else "Entity"


def needs_backtick(label: str) -> bool:
    """Return True if the label/type should be wrapped in backticks."""
    # Backtick if empty, starts with number, or contains special chars
    if not label:
        return True
    if label[0].isdigit():
        return True
    if re.search(r"[^\w\u4e00-\u9fff]", label):
        return True
    return False


def fmt_label(label: str) -> str:
    """Format a label/type for Cypher, adding backticks if needed."""
    safe = sanitize_label(label)
    if needs_backtick(safe):
        return f"`{safe}`"
    return safe


# ============================================================================
# Main conversion
# ============================================================================
def convert():
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        graph = json.load(f)

    graph_name = graph.get("graph_name", "知识图谱")
    entity_list = graph.get("entity_list", [])
    relation_list = graph.get("relation_list", [])
    triple_list = graph.get("triple_list", [])

    entity_map = {e["entity_id"]: e for e in entity_list}
    relation_map = {r["rel_id"]: r for r in relation_list}

    lines = []

    # ----- Header / metadata comment block -----
    lines.append("// ======================================================================")
    lines.append("// Neo4j Database Dump File (.neo4jdb)")
    lines.append(f"// Graph Name: {graph_name}")
    lines.append(f"// Entities:   {len(entity_list)}")
    lines.append(f"// Relations:  {len(relation_list)}")
    lines.append(f"// Triples:    {len(triple_list)}")
    lines.append("// Format:     Cypher script (load via `cypher-shell -f <file>`)")
    lines.append("// ======================================================================")
    lines.append("")
    lines.append("// ----- Clean existing data (idempotent reload) -----")
    lines.append("MATCH (n) DETACH DELETE n;")
    lines.append("")

    # ----- Create uniqueness constraints for fast MERGE & data integrity -----
    lines.append("// ----- Constraints -----")
    # Constraint on Entity nodes by entity_id
    lines.append(
        "CREATE CONSTRAINT entity_id_unique IF NOT EXISTS "
        "FOR (e:Entity) REQUIRE e.entity_id IS UNIQUE;"
    )
    lines.append("")

    # ----- Create nodes -----
    lines.append("// ----- Nodes (Entities) -----")
    for e in entity_list:
        eid = escape_cypher_string(e["entity_id"])
        ename = escape_cypher_string(e["entity_name"])
        etype = escape_cypher_string(e["entity_type"])
        label = fmt_label(e["entity_type"])
        # Every node gets the generic :Entity label plus its specific type label
        # Using MERGE for idempotent reloads
        cypher = (
            f"MERGE (n:Entity:{label} {{entity_id: '{eid}'}}) "
            f"SET n.entity_name = '{ename}', "
            f"n.entity_type = '{etype}';"
        )
        lines.append(cypher)
    lines.append("")

    # ----- Create relationships -----
    lines.append("// ----- Relationships (Triples) -----")
    for t in triple_list:
        sub_id = escape_cypher_string(t["sub_id"])
        rel_id = escape_cypher_string(t["rel_id"])
        obj_id = escape_cypher_string(t["obj_id"])

        rel = relation_map.get(rel_id, {})
        rel_name = rel.get("rel_name", rel_id)
        rel_desc = rel.get("rel_desc", "")
        rel_type = fmt_label(rel_name)

        sub = entity_map.get(sub_id, {})
        obj = entity_map.get(obj_id, {})
        sub_name = escape_cypher_string(sub.get("entity_name", ""))
        obj_name = escape_cypher_string(obj.get("entity_name", ""))

        rel_desc_esc = escape_cypher_string(rel_desc)

        # MATCH both endpoints then CREATE the typed relationship with properties
        cypher = (
            f"MATCH (a:Entity {{entity_id: '{sub_id}'}}), "
            f"(b:Entity {{entity_id: '{obj_id}'}}) "
            f"CREATE (a)-[r:{rel_type} {{"
            f"rel_id: '{rel_id}', "
            f"rel_name: '{escape_cypher_string(rel_name)}', "
            f"rel_desc: '{rel_desc_esc}', "
            f"subject: '{sub_name}', "
            f"object: '{obj_name}'"
            f"}}]->(b);"
        )
        lines.append(cypher)
    lines.append("")

    # ----- Verification queries -----
    lines.append("// ----- Verification -----")
    lines.append(f"// Expected: {len(entity_list)} nodes, {len(triple_list)} relationships")
    lines.append("MATCH (n) RETURN count(n) AS node_count;")
    lines.append("MATCH ()-[r]->() RETURN count(r) AS rel_count;")
    lines.append("MATCH (n) RETURN labels(n) AS labels, count(n) AS cnt ORDER BY cnt DESC;")
    lines.append("")

    # Build a graph meta-node storing the graph_name (optional, useful for provenance)
    lines.append("// ----- Graph metadata node -----")
    lines.append(
        f"MERGE (g:GraphMeta {{graph_name: '{escape_cypher_string(graph_name)}'}}) "
        f"SET g.entity_count = {len(entity_list)}, "
        f"g.relation_count = {len(relation_list)}, "
        f"g.triple_count = {len(triple_list)}, "
        f"g.source_format = 'knowledge_graph_json';"
    )
    lines.append("")

    # ----- Write output -----
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    content = "\n".join(lines)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"Neo4j database file generated: {OUTPUT_PATH}")
    print(f"  Nodes (entities):     {len(entity_list)}")
    print(f"  Relationships:        {len(triple_list)}")
    print(f"  Cypher statements:    {len(lines)} lines")
    print(f"  File size:            {os.path.getsize(OUTPUT_PATH)} bytes")


if __name__ == "__main__":
    convert()
