import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
from sklearn.cluster import DBSCAN

def load_path_data(csv_file):
    """CSVファイルから経路データを読み込み"""
    df = pd.read_csv(csv_file)
    return df['x'].values, df['y'].values

def detect_path_segments(x, y, eps=2.0, min_samples=2):
    """
    経路を連続したセグメントに分割
    急激な方向転換や離散した点を検出してセグメントを分ける
    """
    points = np.column_stack((x, y))
    
    # 連続する点間の距離を計算
    distances = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
    
    # 大きなジャンプを検出（セグメントの境界）
    large_jumps = distances > np.percentile(distances, 90)
    
    # セグメントの開始点を特定
    segment_starts = [0]
    for i, is_jump in enumerate(large_jumps):
        if is_jump:
            segment_starts.append(i + 1)
    segment_starts.append(len(x))
    
    # セグメントに分割
    segments = []
    for i in range(len(segment_starts) - 1):
        start_idx = segment_starts[i]
        end_idx = segment_starts[i + 1]
        if end_idx - start_idx > 1:  # 最低2点必要
            segments.append((x[start_idx:end_idx], y[start_idx:end_idx]))
    
    return segments

def interpolate_segment(x_seg, y_seg, num_points=100, smoothing=0):
    """
    個別セグメントをスプライン補間
    エッジの形状を維持するため、低いスムージングを使用
    """
    if len(x_seg) < 3:
        # 点が少ない場合は線形補間
        t_orig = np.linspace(0, 1, len(x_seg))
        t_new = np.linspace(0, 1, num_points)
        x_interp = np.interp(t_new, t_orig, x_seg)
        y_interp = np.interp(t_new, t_orig, y_seg)
        return x_interp, y_interp
    
    try:
        # パラメータ化されたスプライン補間
        # k=1 (線形) を使用してエッジを保持
        tck, u = splprep([x_seg, y_seg], s=smoothing, k=1)
        u_new = np.linspace(0, 1, num_points)
        x_interp, y_interp = splev(u_new, tck)
        return x_interp, y_interp
    except:
        # スプライン補間が失敗した場合は線形補間
        t_orig = np.linspace(0, 1, len(x_seg))
        t_new = np.linspace(0, 1, num_points)
        x_interp = np.interp(t_new, t_orig, x_seg)
        y_interp = np.interp(t_new, t_orig, y_seg)
        return x_interp, y_interp

def interpolate_path(csv_file, points_per_segment=50, output_file='interpolated_path.csv'):
    """
    メイン関数：経路をインターポレートしてCSVファイルに保存
    """
    # データ読み込み
    x_orig, y_orig = load_path_data(csv_file)
    print(f"元データ: {len(x_orig)} 点")
    
    # セグメントに分割
    segments = detect_path_segments(x_orig, y_orig)
    print(f"検出されたセグメント数: {len(segments)}")
    
    # 各セグメントを補間
    x_interpolated = []
    y_interpolated = []
    
    for i, (x_seg, y_seg) in enumerate(segments):
        print(f"セグメント {i+1}: {len(x_seg)} 点")
        
        # セグメントの長さに応じて補間点数を調整
        segment_length = np.sum(np.sqrt(np.diff(x_seg)**2 + np.diff(y_seg)**2))
        adaptive_points = max(int(segment_length / 2), 10)  # 2mごとに1点、最低10点
        
        x_interp, y_interp = interpolate_segment(x_seg, y_seg, adaptive_points)
        
        x_interpolated.extend(x_interp)
        y_interpolated.extend(y_interp)
    
    # 結果をDataFrameに変換
    result_df = pd.DataFrame({
        'x': x_interpolated,
        'y': y_interpolated
    })
    
    # CSVファイルに保存
    result_df.to_csv(output_file, index=False)
    print(f"補間結果を '{output_file}' に保存しました")
    print(f"補間後のデータ: {len(x_interpolated)} 点")
    
    return np.array(x_interpolated), np.array(y_interpolated)

def visualize_interpolation(csv_file):
    """
    元データと補間データを可視化
    """
    # 元データ
    x_orig, y_orig = load_path_data(csv_file)
    
    # 補間データ
    x_interp, y_interp = interpolate_path(csv_file)
    
    # プロット
    plt.figure(figsize=(12, 8))
    plt.plot(x_orig, y_orig, 'ro-', label='元データ', markersize=3, alpha=0.7)
    plt.plot(x_interp, y_interp, 'b-', label='補間データ', alpha=0.8, linewidth=1)
    
    plt.xlabel('X座標 (m)')
    plt.ylabel('Y座標 (m)')
    plt.title('経路データのインターポレーション（エッジ形状維持）')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.show()

# 使用例
if __name__ == "__main__":
    # CSVファイルのインターポレーション実行
    csv_filename = 'path.csv'  # 入力ファイル名
    
    try:
        # インターポレーション実行
        x_interp, y_interp = interpolate_path(
            csv_filename, 
            points_per_segment=50,
            output_file='interpolated_path.csv'
        )
        
        # 可視化
        visualize_interpolation(csv_filename)
        
        print("\n--- 統計情報 ---")
        print(f"X座標範囲: {np.min(x_interp):.2f} ~ {np.max(x_interp):.2f} m")
        print(f"Y座標範囲: {np.min(y_interp):.2f} ~ {np.max(y_interp):.2f} m")
        print(f"総経路長: {np.sum(np.sqrt(np.diff(x_interp)**2 + np.diff(y_interp)**2)):.2f} m")
        
    except FileNotFoundError:
        print(f"エラー: ファイル '{csv_filename}' が見つかりません")
    except Exception as e:
        print(f"エラーが発生しました: {e}")