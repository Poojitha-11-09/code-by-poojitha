from pyvis.network import Network
import networkx as nx
import numpy as np
import pandas as pd

def create_graph(df, final_preds, threshold=None):
    import numpy as np
    import networkx as nx
    from pyvis.network import Network
    import pandas as pd

    G = nx.Graph()

    transaction_ids = df.index.tolist()
    feature_names = df.columns[1:-1]  # Exclude ID & Transaction_Amount
    features = df.iloc[:, 1:-1].values

    # Z-score computation
    feature_means = df.iloc[:, 1:-1].mean()
    feature_stds = df.iloc[:, 1:-1].std()

    # Unusual feature set
    unusual_features = {
        "feat1", "feat3", "feat12", "Transaction_Amount", "feat14",
        "feat7", "feat10", "feat16", "feat11", "feat2", "feat23",
        "feat20", "feat8", "feat27", "feat21"
    }

    # Step 1: Add Feature Nodes
    for fname in feature_names:
        G.add_node(fname, label=fname, color="orange" if fname in unusual_features else "gray", size=15, title=f"Feature: {fname}")

    # Step 2: Add Transaction Nodes and Edges
    for i, tid in enumerate(transaction_ids):
        is_fraud = final_preds[i]
        node_id = f"Tx {tid}"
        color = "#FF5733" if is_fraud == 1 else "#3498DB"
        connected_features = []

        for j, fname in enumerate(feature_names):
            value = pd.to_numeric(features[i, j], errors='coerce')
            if pd.notna(value):
                mean = feature_means[j]
                std = feature_stds[j]
                z_score = abs((value - mean) / std) if std > 0 else 0

                if is_fraud == 1:
                    if fname in unusual_features and z_score > 0.1:
                        G.add_edge(node_id, fname, weight=z_score, color="#ff3333")  # Red
                        connected_features.append(fname)
                else:
                    if fname not in unusual_features and z_score > 0.1:
                        G.add_edge(node_id, fname, weight=z_score, color="#cccccc")  # Gray only
                        connected_features.append(fname)

        inferred_status = "Fraud" if any(f in unusual_features for f in connected_features) else "Non-Fraud"
        title = f"{node_id}<br>Status: {'Fraud' if is_fraud == 1 else 'Non-Fraud'}<br>Inference: {inferred_status}"
        G.add_node(node_id, label=node_id, color=color, size=25, title=title)

    # Step 3: Create PyVis Network
    net = Network(height="100vh", width="100vw", bgcolor="#000000", font_color="white")
    net.from_nx(G)

    for edge in net.edges:
        edge["width"] = 2

    net.force_atlas_2based(gravity=-50, central_gravity=0.005, spring_length=200, spring_strength=0.1)

    # Step 4: Legend
    legend_html = """
    <div style="position:absolute; top:10px; left:10px; background-color:#111; color:white; 
                padding:10px; border-radius:10px; font-size:14px; z-index:1000;">
        <b>Legend:</b><br>
        🔵 Non-Fraudulent Transaction<br>
        🔴 Fraudulent Transaction<br>
        🟠 Unusual Feature Node<br>
        ⚪️ Normal Feature Node<br>
        🧩 Edge Color = Transactional Deviation<br>
        <svg height='10' width='80'><line x1='0' y1='5' x2='80' y2='5' style='stroke:#cccccc;stroke-width:2' /></svg> (Gray) All Non-Fraud Edges<br>
        <svg height='10' width='80'><line x1='0' y1='5' x2='80' y2='5' style='stroke:#ff3333;stroke-width:2' /></svg> (Red) Fraud Edge to Unusual Feature<br>
    </div>
    """

    # Step 5: Save graph and inject legend
    graph_path = "static/credit_graph.html"
    net.save_graph(graph_path)

    with open(graph_path, "r", encoding="utf-8") as file:
        html = file.read()

    html = html.replace("<body>", f"<body>{legend_html}", 1)

    with open(graph_path, "w", encoding="utf-8") as file:
        file.write(html)

    return graph_path




def create_bitcoin_graph(elliptic_dataset, df_merge):
    G = nx.Graph()
    transaction_ids = range(elliptic_dataset.num_nodes)  # Nodes are already indexed
    fraud_labels = df_merge["Prediction"].apply(lambda x: 1 if x == "Illicit" else 0).values

    # **Step 1: Add Transaction Nodes**
    for i in transaction_ids:
        color = "red" if fraud_labels[i] == 1 else "blue"  # Fraud = Red, Non-Fraud = Blue
        G.add_node(f"Tx {i}", label=f"Tx {i}", color=color, size=25)

    # **Step 2: Add Feature Nodes**
    feature_names = [f"Feature {j}" for j in range(elliptic_dataset.num_features)]
    for fname in feature_names:
        G.add_node(fname, label=fname, color="orange", size=15)

    # **Step 3: Add Transaction ↔ Feature Edges**
    node_features = elliptic_dataset.x.numpy()
    for i, tid in enumerate(transaction_ids):
        for j in range(len(feature_names)):
            weight = float(abs(node_features[i, j]))  # Convert to Python float
            if weight > 0.1:
                G.add_edge(f"Tx {i}", feature_names[j], weight=weight)

    # **Step 4: Add Transaction ↔ Transaction Edges**
    edge_index = elliptic_dataset.edge_index.numpy().T
    for edge in edge_index:
        G.add_edge(f"Tx {edge[0]}", f"Tx {edge[1]}", color="gray", width=float(2))

    # **Step 5: Create PyVis Network**
    net = Network(height="100vh", width="100vw", bgcolor="#000000", font_color="white")

    # **Step 6: Load NetworkX Graph**
    net.from_nx(G)

    # **Step 7: Enable Physics for Better Layout**
    net.force_atlas_2based(gravity=-50, central_gravity=0.005, spring_length=200, spring_strength=0.1)
    graph_path = "static/bitcoin_graph.html"
    net.save_graph(graph_path)
    legend_html = """
    <div style="position: absolute; top: 10px; left: 10px; background-color: rgba(17,17,17,0.8); color: white; 
            padding: 10px; border-radius: 10px; font-size: 14px; z-index: 9999;">
    <b>Legend:</b><br>
    🔵 Non-Fraudulent Transaction<br>
    🔴 Fraudulent Transaction<br>
    🟠 Feature Node
    </div>
    """
    with open(graph_path, "r", encoding="utf-8") as f:
        html_content = f.read()

# Inject the legend just after <body>
    html_content = html_content.replace("<body>", f"<body>{legend_html}")
    with open(graph_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    return graph_path
