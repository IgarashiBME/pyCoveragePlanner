import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# CSVファイルからデータを読み取る
df = pd.read_csv('../interpolated_path_optimized.csv')

# x座標とy座標のデータを抽出する
x = df.iloc[:, 0]
y = df.iloc[:, 1]

# アニメーションの初期化
fig, ax = plt.subplots()
line, = ax.plot([], [], 'b-')  # 軌跡を線で表現する
current_point, = ax.plot([], [], '*', markersize=14, color="red")  # 再生直近の点に星形のマーカーを追加する
ax.set_xlim(min(x), max(x))
ax.set_ylim(min(y), max(y))
ax.set_title('Path Animation')

# アニメーションの更新
def update(frame):
    # 現在の点に星形のマーカーを表示する
    current_point.set_data(x[frame], y[frame])
    # 軌跡を線で表現する
    line.set_data(x[:frame+1], y[:frame+1])
    return line, current_point,

# アニメーションの生成
interval = 1  # アニメーションのインターバルを設定する（ミリ秒）
ani = FuncAnimation(fig, update, frames=len(x), interval=interval, blit=True)
ani.save('animation.gif', writer='pillow')

# アニメーションの再生
plt.show()