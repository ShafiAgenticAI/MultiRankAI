from src.agent.rag_agent import rag_graph

graph_image = rag_graph.get_graph().draw_mermaid_png()

with open(
    "rag_graph.png",
    "wb",
) as f:
    f.write(graph_image)
