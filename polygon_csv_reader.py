import pandas as pd
import numpy as np

def read_outer_boundary_csv(csv_file_path):
    """
    外周境界のCSVファイルを読み取る
    
    :param csv_file_path: CSVファイルのパス
    :return: [(x, y), ...] 形式の座標リスト
    """
    try:
        # CSVファイルを読み込み
        df = pd.read_csv(csv_file_path)
        
        # x, y列が存在するかチェック
        if 'x' not in df.columns or 'y' not in df.columns:
            raise ValueError("CSVファイルに'x'または'y'列が見つかりません")
        
        # 座標リストに変換
        outer_boundary = [(row['x'], row['y']) for _, row in df.iterrows()]
        
        print(f"外周境界: {len(outer_boundary)}点を読み込みました")
        return outer_boundary
        
    except Exception as e:
        print(f"外周境界CSVの読み込みエラー: {e}")
        return []

def read_obstacles_csv(csv_file_path):
    """
    障害物のCSVファイルを読み取る
    
    :param csv_file_path: CSVファイルのパス
    :return: [[(x,y), ...], ...] 形式の障害物リスト
    """
    try:
        # CSVファイルを読み込み
        df = pd.read_csv(csv_file_path)
        
        # 必要な列が存在するかチェック
        required_columns = ['x', 'y', 'index']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"CSVファイルに'{col}'列が見つかりません")
        
        # obstacle_indexでグループ化
        grouped = df.groupby('index')
        
        obstacles = []
        for obstacle_index, group in grouped:
            # 各障害物の座標リストを作成
            obstacle_coords = [(row['x'], row['y']) for _, row in group.iterrows()]
            obstacles.append(obstacle_coords)
            print(f"障害物{obstacle_index}: {len(obstacle_coords)}点を読み込みました")
        
        print(f"総障害物数: {len(obstacles)}個")
        return obstacles
        
    except Exception as e:
        print(f"障害物CSVの読み込みエラー: {e}")
        return []

"""if __name__ == '__main__':
    # CSVファイルから読み込み
    outer_boundary = read_outer_boundary_csv('csv/ichiko.csv')
    obstacles = read_obstacles_csv('csv/obstacles.csv')
    
    # 読み込み結果を確認
    print(f"\n読み込み結果:")
    print(f"外周境界: {outer_boundary}")
    print(f"障害物: {obstacles}")"""