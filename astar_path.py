import heapq
import numpy as np
from shapely.geometry import Point, LineString, Polygon

def find_path(start, end, drivable_area, all_hole_polygons):
    """
    A*アルゴリズムを使用して、走行可能領域内を通過する最短経路を見つける。

    Args:
        start (tuple): 開始地点の座標 (x, y)
        end (tuple): 終了地点の座標 (x, y)
        drivable_area (Polygon): 外周から障害物をくり抜いた走行可能領域のShapely Polygon
        all_hole_polygons (list): 障害物ポリゴンのリスト (Shapely Polygon)

    Returns:
        list or None: 経路上の点のリスト。見つからない場合はNone。
    """
    
    # --- 1. グラフのノードを定義 ---
    # ノード = スタート + ゴール + 外周の頂点 + 全ての障害物の頂点
    all_vertices = [start, end]
    all_vertices.extend(list(drivable_area.exterior.coords))
    for poly in all_hole_polygons:
        all_vertices.extend(list(poly.exterior.coords))
    
    # 重複を除去し、辞書を作成
    nodes = list(set(all_vertices))
    node_map = {i: coord for i, coord in enumerate(nodes)}
    coord_map = {coord: i for i, coord in node_map.items()}

    start_node = coord_map[start]
    end_node = coord_map[end]

    # --- 2. 可視性グラフを構築 ---
    adjacency = {i: [] for i in range(len(nodes))}
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            p1 = node_map[i]
            p2 = node_map[j]
            line = LineString([p1, p2])

            # ### 修正点: 走行可能領域内にあるかをチェック ###
            # 線分の中点が走行可能領域内にあるか、かつ、
            # 線分が領域と交差していない（領域外に出ていない）かをチェック。
            # buffer(1e-9)は浮動小数点演算の誤差を吸収するおまじない。
            if drivable_area.buffer(1e-9).contains(line):
                dist = np.linalg.norm(np.array(p1) - np.array(p2))
                adjacency[i].append((j, dist))
                adjacency[j].append((i, dist))

    # --- 3. A*アルゴリズムの実行 (この部分は変更なし) ---
    open_set = [(0, start_node)]
    came_from = {}
    
    g_score = {i: float('inf') for i in range(len(nodes))}
    g_score[start_node] = 0
    
    f_score = {i: float('inf') for i in range(len(nodes))}
    f_score[start_node] = np.linalg.norm(np.array(start) - np.array(end))

    while open_set:
        _, current_node = heapq.heappop(open_set)

        if current_node == end_node:
            path = []
            while current_node in came_from:
                path.append(node_map[current_node])
                current_node = came_from[current_node]
            path.append(start)
            return path[::-1]

        for neighbor_node, distance in adjacency[current_node]:
            tentative_g_score = g_score[current_node] + distance
            if tentative_g_score < g_score[neighbor_node]:
                came_from[neighbor_node] = current_node
                g_score[neighbor_node] = tentative_g_score
                h = np.linalg.norm(np.array(node_map[neighbor_node]) - np.array(end))
                f_score[neighbor_node] = tentative_g_score + h
                if not any(n_id == neighbor_node for _, n_id in open_set):
                    heapq.heappush(open_set, (f_score[neighbor_node], neighbor_node))
    
    return None