import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as PolygonPatch
import numpy as np
import json

# --- モジュールのインポート ---
import survey_path
import polygon_csv_reader as pcr
import astar_path

import itertools
import networkx as nx
from shapely.geometry import Polygon, LineString, Point as ShapelyPoint
from shapely.ops import unary_union

SPACING = 0.8
ANGLE_DEG = 90


# --- データ構造とヘルパー関数 ---
# Point, Edge, Trapezoid, DecompositionResult, PolygonCell,
# GeneralizedDecompositionResult, point_in_polygon, rotate_points,
# trapezoidal_decomposition, decomposition_with_angle
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class Edge:
    def __init__(self, p1, p2):
        if p1.x < p2.x:
            self.p1 = p1
            self.p2 = p2
        else:
            self.p1 = p2
            self.p2 = p1

        if self.p1.x == self.p2.x:
            raise ValueError("Vertical edges are not supported in this simple implementation.")

    def y_at_x(self, x):
        if x < self.p1.x or x > self.p2.x:
            return None

        # 垂直な辺がないため、ゼロ除算の心配はない
        t = (x - self.p1.x) / (self.p2.x - self.p1.x)
        return self.p1.y + t * (self.p2.y - self.p1.y)


class Trapezoid:
    def __init__(self, left_x, right_x, top_edge, bottom_edge, trapezoid_id=None):
        self.left_x = left_x
        self.right_x = right_x
        self.top_edge = top_edge
        self.bottom_edge = bottom_edge
        self.id = trapezoid_id

    def get_corners(self):
        return [
            (self.left_x, self.bottom_edge.y_at_x(self.left_x)),
            (self.right_x, self.bottom_edge.y_at_x(self.right_x)),
            (self.right_x, self.top_edge.y_at_x(self.right_x)),
            (self.left_x, self.top_edge.y_at_x(self.left_x)),
        ]

    def get_vertices_array(self):
        corners = self.get_corners()
        return np.array(corners)

    def get_center(self):
        center_x = (self.left_x + self.right_x) / 2
        center_y = (self.top_edge.y_at_x(center_x) + self.bottom_edge.y_at_x(center_x)) / 2
        return (center_x, center_y)


class DecompositionResult:
    def __init__(self):
        self.trapezoids = []
        self.total_count = 0

    def add_trapezoid(self, trapezoid):
        trapezoid.id = self.total_count + 1
        self.trapezoids.append(trapezoid)
        self.total_count += 1


class PolygonCell:
    def __init__(self, vertices, cell_id=None):
        self.vertices, self.id = vertices, cell_id

    def get_corners(self):
        return self.vertices

    def get_vertices_array(self):
        return np.array(self.vertices)

    def get_center(self):
        return (sum(v[0] for v in self.vertices) / len(self.vertices), sum(v[1] for v in self.vertices) / len(self.vertices)) if self.vertices else (0,0)


class GeneralizedDecompositionResult:
    def __init__(self):
        self.cells, self.total_count = [], 0

    def add_cell(self, cell):
        cell.id = len(self.cells) + 1
        self.cells.append(cell)
        self.total_count = len(self.cells)


def point_in_polygon(point, polygon_points):
    """点の内外判定（レイキャスティング法）"""
    x, y = point.x, point.y
    n = len(polygon_points)
    inside = False

    p1x, p1y = polygon_points[0]
    for i in range(n + 1):
        p2x, p2y = polygon_points[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside


def rotate_points(points, angle_rad):
    """点のリストを指定された角度（ラジアン）で回転させる"""
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    rotated = []
    for x, y in points:
        new_x = x * cos_a - y * sin_a
        new_y = x * sin_a + y * cos_a
        rotated.append((new_x, new_y))
    return rotated


def trapezoidal_decomposition(outer_polygon, holes=[]):
    """    凹型多角形（障害物あり）を垂直な線で台形に分割する    """
    all_vertices = []
    all_edges = []
    result = DecompositionResult()

    polygons = [outer_polygon] + holes
    for poly_coords in polygons:
        try:
            poly_points = [Point(x, y) for x, y in poly_coords]
            all_vertices.extend(poly_points)
            for i in range(len(poly_points)):
                p1 = poly_points[i]
                p2 = poly_points[(i + 1) % len(poly_points)]
                if p1.x != p2.x:
                    all_edges.append(Edge(p1, p2))
        except ValueError as e:
            # 垂直な辺は回転後の座標系で発生しうるので、ここでハンドルする
            # print(f"Skipping vertical edge after rotation: {e}")
            pass # 垂直な辺は無視して続行

    sorted_vertices = sorted(all_vertices, key=lambda p: p.x)

    if not sorted_vertices:
        return result

    unique_x_coords = sorted(list(set(p.x for p in sorted_vertices)))

    for i in range(len(unique_x_coords) - 1):
        left_x = unique_x_coords[i]
        right_x = unique_x_coords[i+1]

        if left_x == right_x: continue # 同じ座標ならスキップ

        mid_x = (left_x + right_x) / 2

        active_edges = []
        for edge in all_edges:
            # エッジがスラブを完全にまたいでいるかチェック
            if edge.p1.x <= left_x and edge.p2.x >= right_x:
                active_edges.append(edge)

        active_edges.sort(key=lambda e: e.y_at_x(mid_x))

        for j in range(len(active_edges) - 1):
            bottom_edge = active_edges[j]
            top_edge = active_edges[j+1]

            trapezoid = Trapezoid(left_x, right_x, top_edge, bottom_edge)

            center_x = (left_x + right_x) / 2
            center_y = (top_edge.y_at_x(center_x) + bottom_edge.y_at_x(center_x)) / 2
            test_point = Point(center_x, center_y)

            if point_in_polygon(test_point, outer_polygon):
                is_in_hole = False
                for hole_poly in holes:
                    if point_in_polygon(test_point, hole_poly):
                        is_in_hole = True
                        break
                if not is_in_hole:
                    result.add_trapezoid(trapezoid)

    return result


def decomposition_with_angle(outer_polygon, holes, angle_deg=0):
    """
    指定された角度でポリゴンを分割する
    :param outer_polygon: 外周ポリゴンの頂点リスト
    :param holes: 障害物ポリゴンのリスト
    :param angle_deg: 分割線の角度（度数法）。0度は垂直、90度は水平。
    :return: GeneralizedDecompositionResult オブジェクト
    """
    # 1. 座標系を回転させる角度を計算（ラジアン）
    # angle_deg=30 で分割したい場合、座標系を -30度 回転させる
    angle_rad = np.deg2rad(angle_deg)
    rotation_rad = -angle_rad

    # 2. 全てのポリゴンを回転
    rotated_outer = rotate_points(outer_polygon, rotation_rad)
    rotated_holes = [rotate_points(hole, rotation_rad) for hole in holes]

    # 3. 回転した座標系で、既存の（垂直な）台形分割アルゴリズムを実行
    # このとき、回転によって垂直になった辺はエラーになる可能性があるが、
    # trapezoidal_decomposition内で処理される。
    temp_result = trapezoidal_decomposition(rotated_outer, rotated_holes)

    # 4. 結果を格納するための新しいオブジェクトを作成
    final_result = GeneralizedDecompositionResult()

    # 5. 生成された各台形の頂点を逆回転させ、新しい結果オブジェクトに格納
    for trap in temp_result.trapezoids:
        final_result.add_cell(PolygonCell(rotate_points(trap.get_corners(), -rotation_rad)))
    return final_result


# --- セルマージ関連の関数 ---
def check_merge_validity(polygon_shapely, survey_path_module):
    """
    【修正】マージ妥当性チェックを強化。
    生成されたパスの各「線分」がポリゴン内に含まれるかチェックする。
    """
    poly_coords_np = np.array(polygon_shapely.exterior.coords)

    try:
        path_points, _ = survey_path_module.calculate(poly_coords_np)
    except Exception:
        return False

    if path_points is None or len(path_points) < 2:
        return True

    # 浮動小数点誤差を考慮してポリゴンをわずかに膨らませる
    buffered_poly = polygon_shapely.buffer(1e-9)

    # 【修正点】点ではなく、線分でチェックする
    for i in range(len(path_points) - 1):
        p1 = path_points[i]
        p2 = path_points[i+1]
        # 2点間の線分を作成
        segment = LineString([p1, p2])
        # 線分がポリゴン内に完全に含まれているかチェック
        if not buffered_poly.contains(segment):
            return False # 1線分でもはみ出したらマージは無効

    return True


def merge_adjacent_cells(initial_cells, survey_path_module):
    """
    【修正】ロジックを微調整し、IDの再割り当てを最後に一括で行うように変更。
    """
    cells = list(initial_cells)

    while True:
        merged_in_this_iteration = False
        best_merge_candidate = None
        max_shared_edge_length = 1e-9

        # i, j を使って直接リストを操作するとインデックスがずれる危険があるため、
        # マージ候補を保持しておき、イテレーションの最後にマージする方式に変更。

        # すべてのセルペアをチェック
        for i in range(len(cells)):
            for j in range(i + 1, len(cells)):
                cell1 = cells[i]
                cell2 = cells[j]

                # try-exceptブロックでshapelyの潜在的なエラーを捕捉
                try:
                    poly1 = Polygon(cell1.get_corners())
                    poly2 = Polygon(cell2.get_corners())

                    if not poly1.is_valid or not poly2.is_valid: continue

                    intersection = poly1.intersection(poly2)
                    if intersection.geom_type == 'LineString' and intersection.length > max_shared_edge_length:
                        merged_shapely_poly = unary_union([poly1, poly2]).buffer(0) # buffer(0)で無効なジオメトリを修正

                        if not isinstance(merged_shapely_poly, Polygon): continue

                        # この時点でマージ妥当性チェックを行う
                        if check_merge_validity(merged_shapely_poly, survey_path_module):
                             # より長い辺を共有するペアを優先
                            if intersection.length > max_shared_edge_length:
                                max_shared_edge_length = intersection.length
                                best_merge_candidate = (i, j, merged_shapely_poly)
                except Exception:
                    continue # Shapelyでエラーが出たらそのペアはスキップ

        if best_merge_candidate:
            i_to_merge, j_to_merge, merged_poly = best_merge_candidate

            new_cell = PolygonCell(list(merged_poly.exterior.coords))

            # 古いセルを削除し、新しいセルを追加 (インデックスが大きい方から削除)
            # マージ対象のセルをNoneでマークし、最後にまとめて削除する方が安全
            cells.append(new_cell)
            cells[j_to_merge] = None # マーク
            cells[i_to_merge] = None # マーク

            cells = [c for c in cells if c is not None] # Noneでないものだけ残す

            merged_in_this_iteration = True

        if not merged_in_this_iteration:
            break

    # 最後にIDを再割り当て
    for idx, cell in enumerate(cells):
        cell.id = idx + 1

    return cells


# --- 【修正】経路生成関連 ---
def build_adjacency_graph(cells, obstacle_polygons):
    """
    【修正】障害物を考慮した隣接グラフ構築。
    中心点を結ぶ線が「走行可能領域」内にあるかで判定。
    """
    G = nx.Graph()
    cell_polygons = {c.id: Polygon(c.get_corners()) for c in cells}

    # 【重要】走行可能領域をここで計算
    outer_shapely = Polygon(pcr.read_outer_boundary_csv('polygon_data/ichiko.csv')) # 外周を再読み込み
    obstacles_union = unary_union([Polygon(obs) for obs in obstacle_polygons]) if obstacle_polygons else Polygon()
    drivable_area = outer_shapely.difference(obstacles_union)

    for cell in cells:
        G.add_node(cell.id, center=cell.get_center())

    for c1, c2 in itertools.combinations(cells, 2):
        poly1 = cell_polygons[c1.id]
        poly2 = cell_polygons[c2.id]

        try:
            # 辺または頂点で接しているか
            if poly1.touches(poly2):
                center1 = c1.get_center()
                center2 = c2.get_center()
                transition_line = LineString([center1, center2])

                # 【修正点】中心間を結ぶ直線が走行可能領域内にあるかチェック
                # buffer(1e-9)は数値誤差対策
                if drivable_area.buffer(1e-9).contains(transition_line):
                    distance = np.linalg.norm(np.array(center1) - np.array(center2))
                    G.add_edge(c1.id, c2.id, weight=distance)
        except Exception:
            continue
    return G


def get_optimal_visit_order(graph):
    """
    貪欲法（最近傍法）を使ってセルの訪問順序を決定する。
    """
    if not graph.nodes:
        return []

    # 最も左上にあるノードを開始点とする
    nodes = list(graph.nodes(data=True))
    start_node = min(nodes, key=lambda n: n[1]['center'][0] + n[1]['center'][1] * -1)[0]

    unvisited = list(graph.nodes)
    unvisited.remove(start_node)

    path = [start_node]
    current_node = start_node

    while unvisited:
        # 現在地から行ける、まだ訪れていないノードを探す
        neighbors = [n for n in graph.neighbors(current_node) if n in unvisited]

        if not neighbors:
            # 袋小路に入った場合：未訪問のノードの中で最も近いノードへジャンプ
            # (より高度な実装ではA*などを使う)
            min_dist = float('inf')
            next_node = None
            current_center = graph.nodes[current_node]['center']
            for node in unvisited:
                dist = np.linalg.norm(np.array(current_center) - np.array(graph.nodes[node]['center']))
                if dist < min_dist:
                    min_dist = dist
                    next_node = node
        else:
            # 隣接ノードの中で最も近いノードを選択
            next_node = min(neighbors, key=lambda n: graph[current_node][n]['weight'])

        path.append(next_node)
        unvisited.remove(next_node)
        current_node = next_node

    return path


def generate_and_connect_paths(ordered_cell_ids, cells_list, outer_polygon_coords, obstacle_coords_list, spacing):
    """
    【全面的な見直し】経路生成と接続のロジックを堅牢化。
    """
    if not ordered_cell_ids:
        return np.array([])

    full_path_with_id = []
    cells_map = {c.id: c for c in cells_list}

    # 走行可能領域と障害物リストをShapelyオブジェクトとして準備
    outer_shapely = Polygon(outer_polygon_coords)
    obstacles_shapely_list = [Polygon(obs) for obs in obstacle_coords_list]
    obstacles_union = unary_union(obstacles_shapely_list) if obstacles_shapely_list else Polygon()
    drivable_area = outer_shapely.difference(obstacles_union)

    previous_path_end_point = None

    for cell_id in ordered_cell_ids:
        # 1. 現在のセルの内部パスを生成
        current_cell = cells_map.get(cell_id)
        if not current_cell: continue

        current_cell_np = current_cell.get_vertices_array()
        inner_path, _ = survey_path.calculate(current_cell_np, spacing)

        if inner_path.shape[0] < 2:
            continue

        # 2. パスの向きを決定 (前のパスの終点に最も近い方を始点にする)
        if previous_path_end_point is not None:
            path_start = inner_path[0]
            path_end = inner_path[-1]
            if np.linalg.norm(previous_path_end_point - path_end) < np.linalg.norm(previous_path_end_point - path_start):
                inner_path = inner_path[::-1]

        current_path_start_point = inner_path[0]

        # 3. 遷移パスを生成 (前のパスの終点から今のパスの始点へ)
        if previous_path_end_point is not None:
            start_trans = tuple(previous_path_end_point)
            end_trans = tuple(current_path_start_point)

            # 直線で結べるかチェック
            transition_line = LineString([start_trans, end_trans])

            # 【重要】走行可能領域(drivable_area)を使って障害物との衝突をチェック
            if drivable_area.buffer(-1e-9).contains(transition_line):
                # 直線でOKなら、その線分を遷移パスとする (始点と終点は含めない)
                # このケースでは点が少ないので、単純に接続するだけでOK
                pass # 遷移パスは不要
            else:
                # 障害物があればA*で経路探索
                # astar_path.find_pathは走行可能領域と障害物リストを引数に取ると仮定
                transition_path = astar_path.find_path(
                    start_trans, end_trans, drivable_area, obstacles_shapely_list
                )
                if transition_path and len(transition_path) > 2:
                    # 遷移パスの点には cell_id = 0 を付与
                    for p in transition_path[1:-1]:
                        full_path_with_id.append([p[0], p[1], 0])

        # 4. セル内パスを追加 (cell_idを付与)
        for p in inner_path:
            full_path_with_id.append([p[0], p[1], cell_id])

        # 5. このパスの終点を次のループのために保存
        previous_path_end_point = inner_path[-1]

    return np.array(full_path_with_id) if full_path_with_id else np.array([])


# --- 可視化関数 ---
def visualize_generalized(outer_polygon, holes, decomposition_result, path=None, title_suffix=""):
    fig, ax = plt.subplots(figsize=(15, 10))

    # ★【修正】描画する前にデータが妥当かチェック
    if outer_polygon and len(outer_polygon) > 2:
        ax.add_patch(PolygonPatch(outer_polygon, facecolor='lightgrey', edgecolor='black', lw=2, alpha=0.5))
    else:
        print("警告: 外周ポリゴンが描画できるほど有効ではありません。")

    for hole in holes:
        if hole and len(hole) > 2:
            ax.add_patch(PolygonPatch(hole, facecolor='white', edgecolor='black', lw=2))

    if decomposition_result and decomposition_result.total_count > 0:
        colors = plt.cm.viridis(np.linspace(0, 1, decomposition_result.total_count))
        for i, cell in enumerate(decomposition_result.cells):
            corners = cell.get_corners()
            ax.add_patch(PolygonPatch(corners, facecolor=colors[i % len(colors)], edgecolor='purple', alpha=0.6, lw=1))
            center_x, center_y = cell.get_center()
            ax.text(center_x, center_y, str(cell.id), fontsize=10, ha='center', va='center',
                    bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.7))

    if path is not None and path.size > 0:
        # パスをcell_idで色分けして描画
        unique_ids = np.unique(path[:, 2])
        path_colors = plt.cm.jet(np.linspace(0, 1, len(unique_ids)))
        id_color_map = {uid: color for uid, color in zip(unique_ids, path_colors)}

        for uid in unique_ids:
            segment = path[path[:, 2] == uid]
            label = f'Cell {int(uid)}' if uid > 0 else 'Transition'
            color = 'red' if uid == 0 else id_color_map[uid]
            lw = 3 if uid == 0 else 1.5
            ax.plot(segment[:, 0], segment[:, 1], color=color, lw=lw, label=label, marker='o', markersize=2, alpha=0.8)

        ax.plot(path[0, 0], path[0, 1], 'go', markersize=12, label="Start", markeredgecolor='k')
        ax.plot(path[-1, 0], path[-1, 1], 'yo', markersize=12, label="End", markeredgecolor='k')

    ax.set_aspect('equal', 'box')
    # ★【修正】描画範囲を自動調整
    ax.autoscale_view()
    plt.title(f"Decomposition and Path {title_suffix}", fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.6)
    # plt.legend() # ラベルが多すぎると凡例が見にくくなるのでコメントアウト
    plt.show()


if __name__ == '__main__':
    # 1. データ読み込み
    outer_boundary = pcr.read_outer_boundary_csv('polygon_data/ichiko.csv')
    obstacles = pcr.read_obstacles_csv('polygon_data/obstacles.csv')

    # ★【修正】データ読み込みの成否をチェック
    if not outer_boundary or len(outer_boundary) < 3:
        print("エラー: 外周データが読み込めませんでした。'csv/ichiko.csv'のパスと内容を確認してください。")
        # データがなくても空のプロットウィンドウを表示してみる
        visualize_generalized(outer_boundary, obstacles, None, title_suffix="(Error: Could not load data)")
        exit()
    print(f"外周データを読み込みました。頂点数: {len(outer_boundary)}")
    print(f"障害物データを読み込みました。障害物数: {len(obstacles)}")

    # 2. 角度を指定して領域を分割
    initial_decomposition = decomposition_with_angle(outer_boundary, obstacles, angle_deg=ANGLE_DEG)
    print(f"初期分割完了。セル数: {initial_decomposition.total_count}")

    # ★【修正】分割直後の状態を可視化（セルがなくても外形は表示する）
    visualize_generalized(outer_boundary, obstacles, initial_decomposition, path=None,
                          title_suffix=f"(Initial Decomposition at {ANGLE_DEG} degrees)")

    if initial_decomposition.total_count > 0:
        # 3. セルのマージ処理
        print("セルのマージ処理を開始します...")
        merged_cells = merge_adjacent_cells(initial_decomposition.cells, survey_path)

        final_result = GeneralizedDecompositionResult()
        for cell in merged_cells:
            final_result.add_cell(cell)

        print(f"マージ完了。セル数: {initial_decomposition.total_count} -> {final_result.total_count}")

        # マージ後の状態を可視化
        visualize_generalized(outer_boundary, obstacles, final_result, path=None,
                              title_suffix="(After Merging Cells)")

        cells = final_result.cells

        # 4. 隣接グラフを構築
        print("セルの隣接グラフを構築中...")
        adjacency_graph = build_adjacency_graph(cells, obstacles)
        print(f"グラフ構築完了。ノード数: {adjacency_graph.number_of_nodes()}, エッジ数: {adjacency_graph.number_of_edges()}")

        # 5. 最適な訪問順序を決定
        print("最適な訪問順序を計算中...")
        optimal_order = get_optimal_visit_order(adjacency_graph)
        print("訪問順序:", optimal_order)

        # 6. 順序に従ってパスを生成・接続
        print("順序に従って最終的なパスを生成中...")
        final_path = generate_and_connect_paths(optimal_order, cells, outer_boundary, obstacles, SPACING)
        print("最終パスの生成完了。")

        # 7. 最終結果を可視化
        visualize_generalized(outer_boundary, obstacles, final_result, path=final_path,
                              title_suffix=f"(Final Path after Merging to {len(optimal_order)} cells)")

        # 8. 結果をファイルに保存
        if final_path.size > 0:
            np.savetxt('path_optimized.csv', final_path, fmt=['%.7f', '%.7f', '%d'], delimiter=',')
            print("マージ・最適化されたパスを path_merged_optimized.csv に保存しました。")

            if hasattr(survey_path, 'interpolate'):
                interpolated_path = survey_path.interpolate(final_path[:, :2])
                np.savetxt('interpolated_path_optimized.csv', interpolated_path, fmt='%.7f', delimiter=',')
                print("補間パスを interpolated_path_merged_optimized.csv に保存しました。")
    else:
        print("領域が生成されませんでした。マージ以降の処理をスキップします。")