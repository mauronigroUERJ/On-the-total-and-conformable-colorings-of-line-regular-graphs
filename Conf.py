import networkx as nx
import pulp
import matplotlib.pyplot as plt
import os

def solve_conformable_coloring(g6_file_path):
    failure_file = "no_solution.txt"
    if os.path.exists(failure_file):
        os.remove(failure_file)

    if not os.path.exists(g6_file_path):
        print(f"Erro: Arquivo {g6_file_path} não encontrado.")
        return

    # Processa o arquivo linha por linha
    with open(g6_file_path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    for idx, g6_string in enumerate(lines):
        try:
            # CORREÇÃO: Converte a string para bytes e usa a função correta
            G = nx.from_graph6_bytes(g6_string.encode('ascii'))
        except Exception as e:
            print(f"Erro ao ler linha {idx} ({g6_string}): {e}")
            continue

        edges = list(G.edges())
        num_edges = len(edges)
        
        if num_edges == 0:
            print(f"Grafo {idx}: Sem arestas. Pulando...")
            continue

        delta = max(dict(G.degree()).values())
        num_colors = 2 * delta - 1
        target_parity = num_edges % 2

        print(f"Processando Grafo {idx} [{g6_string}]... ", end="")

        if num_colors <= 0:
            registrar_falha(failure_file, g6_string)
            print("Falha (Delta=0)")
            continue

        # --- MODELO PLI ---
        prob = pulp.LpProblem(f"Coloring_{idx}", pulp.LpMinimize)
        x = pulp.LpVariable.dicts("x", (range(num_edges), range(num_colors)), cat='Binary')
        y = pulp.LpVariable.dicts("y", range(num_colors), lowBound=0, cat='Integer')

        for i in range(num_edges):
            prob += pulp.lpSum([x[i][c] for c in range(num_colors)]) == 1

        for v in G.nodes():
            adj_edges_idx = [i for i, e in enumerate(edges) if v in e]
            for c in range(num_colors):
                prob += pulp.lpSum([x[i][c] for i in adj_edges_idx]) <= 1

        for c in range(num_colors):
            prob += pulp.lpSum([x[i][c] for i in range(num_edges)]) == 2 * y[c] + target_parity

        status = prob.solve(pulp.PULP_CBC_CMD(msg=0))

        if pulp.LpStatus[status] == 'Optimal':
            salvar_imagem_combinada(G, edges, x, num_colors, target_parity, idx, g6_string)
            print("Sucesso! Imagem gerada.")
        else:
            registrar_falha(failure_file, g6_string)
            print("Sem solução.")

def registrar_falha(path, g6_str):
    with open(path, "a") as f:
        f.write(f"{g6_str}\n")

def salvar_imagem_combinada(G, edges, x_vars, num_colors, target_parity, idx, g6_str):
    color_assignment = {}
    color_classes = {c: 0 for c in range(num_colors)}
    
    table_data = []
    for i, e in enumerate(edges):
        for c in range(num_colors):
            if pulp.value(x_vars[i][c]) == 1:
                color_assignment[e] = c
                color_classes[c] += 1
                table_data.append([f"{e[0]}-{e[1]}", c])

    # Criar a figura com dois subplots (Grafo | Tabela)
    fig, (ax_graph, ax_table) = plt.subplots(1, 2, figsize=(14, 7), gridspec_kw={'width_ratios': [1.2, 1]})
    
    # 1. Desenhar o Grafo
    pos = nx.circular_layout(G)
    edge_colors = [color_assignment[e] for e in G.edges()]
    cmap = plt.get_cmap('rainbow', num_colors)
    
    nx.draw_networkx_nodes(G, pos, ax=ax_graph, node_size=500, node_color='lightgray', edgecolors='black')
    nx.draw_networkx_labels(G, pos, ax=ax_graph, font_size=10, font_weight='bold')
    
    edges_plot = nx.draw_networkx_edges(
        G, pos, ax=ax_graph, edge_color=edge_colors, 
        width=3, edge_cmap=cmap, edge_vmin=0, edge_vmax=num_colors-1
    )
    
    ax_graph.set_title(f"Grafo {idx}: {g6_str}\n$2\\Delta-1$ Coloração (Paridade: {target_parity})", fontsize=12)
    ax_graph.axis('off')

    # 2. Criar a Tabela de Cores e Paridade
    ax_table.axis('off')
    
    # Preparar dados da verificação de paridade para a tabela
    parity_summary = [[f"Cor {c}", f"{count}", "OK" if count % 2 == target_parity else "ERRO"] 
                      for c, count in color_classes.items()]
    
    # Tabela principal (Arestas)
    header = ["Aresta", "Cor"]
    the_table = ax_table.table(cellText=table_data, colLabels=header, loc='center', cellLoc='center')
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(9)
    the_table.scale(1, 1.2)

    # Tabela de Resumo (abaixo da principal)
    parity_table = ax_table.table(cellText=parity_summary, colLabels=["Classe", "Tam.", "Paridade"], 
                                  loc='bottom', cellLoc='center', bbox=[0, -0.4, 1, 0.3])
    parity_table.auto_set_font_size(False)
    parity_table.set_fontsize(9)

    plt.tight_layout()
    # Salva o PNG
    filename = f"resultado_grafo_{idx}.png"
    plt.savefig(filename, dpi=150)
    plt.close()

if __name__ == "__main__":
    solve_conformable_coloring("graphs.g6")
