# インポート
import numpy as np
import pandas as pd
from copy import deepcopy
import queue
import random

# データの読み込みとネットワークの認識
nodelist = [  # 与えられたノードの設定 送信ノード番号・受信ノード番号・距離・容量
    [0, 1, 1, 3],
    [0, 3, 1, 3],
    [1, 2, 1, 3],
    [1, 3, 1, 4],
    [2, 4, 1, 4],
    [2, 5, 1, 3],
    [3, 4, 1, 5],
    [3, 7, 1, 3],
    [4, 5, 1, 5],
    [4, 7, 1, 4],
    [5, 6, 1, 3],
    [5, 8, 1, 4],
    [6, 8, 1, 3],
    [7, 8, 1, 4],
    [7, 9, 1, 3],
    [8, 9, 1, 3],
]


def calcNetwork(nodelist):
    array_size = max(nodelist[-1][0], nodelist[-1][1]) + 1
    nw = np.zeros((array_size, array_size), dtype=np.int)
    for node in nodelist:
        node_idx_1 = node[0]
        node_idx_2 = node[1]
        link_weight = node[3]
        nw[node_idx_1][node_idx_2] = link_weight
        nw[node_idx_2][node_idx_1] = link_weight
    return nw


network_shape = calcNetwork(nodelist)

# 通信を行う関数の定義
"""
10,000回通信を行う。

algolithm : 'min_hop' | 'max_thp' 経路選択アルゴリズム
c_type : 'fixed' | 'each' 固定経路方式か要求時経路選択方式か
n : 同時並行で実行されるconnectionの数
N : 通信を行う合計回数
network_shape : ネットワークの構造
"""


def make10kconnection(algolithm, c_type, n, N, network_shape):
    current_network = deepcopy(network_shape)  # ネットワークの状態を保持する
    number_of_nodes = current_network.shape[0]
    # 固定経路方式全てのノード間の経路を計算する
    if c_type == "fixed":
        routeTable = np.empty(network_shape.shape, dtype=np.object)
        for idx_i in range(number_of_nodes):
            for idx_j in range(number_of_nodes):
                routeTable[idx_i][idx_j] = (
                    routeMinHop(idx_i, idx_j, current_network)
                    if algolithm == "min_hop"
                    else routeMaxThp(idx_i, idx_j, current_network)
                )

    connection_queue = queue.Queue()  # 通信を管理するキュー
    loss_count = 0  # 通信できなかった回数
    for k in range(N + n):
        # ランダムに送受信ノードi,jを決定する
        ns = []
        while len(ns) < 2:
            num = random.randint(0, number_of_nodes - 1)
            if num not in ns:
                ns.append(num)
        i = ns[0]
        j = ns[1]
        # i,j間の経路を取得する
        if c_type == "fixed":  # 固定経路方式の場合、事前に計算したrouteTableから取得する
            route = routeTable[i][j]
        elif c_type == "each":  # 要求時経路の場合、routeを計算
            route = (
                routeMinHop(i, j, current_network)
                if algolithm == "min_hop"
                else routeMaxThp(i, j, current_network)
            )

        if k < N:
            # 経路上の空き容量のチェック
            zero_route_flag = False
            if route:  # routeが存在すれば、リンク容量が存在するかチェック
                for start_node_idx in range(len(route) - 1):
                    end_node_idx = start_node_idx + 1
                    if current_network[route[start_node_idx]][route[end_node_idx]] == 0:
                        zero_route_flag = True
            else:  # routeが存在しない場合
                zero_route_flag = True
            # 空き容量があれば通信を行う
            if zero_route_flag:
                connection_queue.put(None)
                loss_count += 1
            else:
                connection_queue.put(route)
                for start_node_idx in range(len(route) - 1):
                    end_node_idx = start_node_idx + 1
                    current_network[route[start_node_idx]][route[end_node_idx]] -= 1
                    current_network[route[end_node_idx]][route[start_node_idx]] -= 1
        # 通信の終了
        if k > n - 1:
            end_route = connection_queue.get()
            if end_route:  # end_routeがNoneでなければ
                for start_node_idx in range(len(end_route) - 1):
                    end_node_idx = start_node_idx + 1
                    current_network[end_route[start_node_idx]][
                        end_route[end_node_idx]
                    ] += 1
                    current_network[end_route[end_node_idx]][
                        end_route[start_node_idx]
                    ] += 1

    return (loss_count / N) * 100  # 呼損率


"""
最小ホップ経路を返す

i,j : ノードのインデックス
network_shape : ネットワーク
"""


def routeMinHop(i, j, current_network):
    # ノードの状態を管理する配列の作成
    number_of_node = current_network.shape[0]
    column_names = ["node", "dist", "chk", "prev"]
    node_state_array = []
    for node_idx in range(number_of_node):
        node_state = [node_idx, float("inf"), 0, None]
        node_state_array.append(node_state)
    node_state_df = pd.DataFrame(node_state_array, columns=column_names)
    # スタートノードの設定
    node_state_df.at[i, "dist"] = 0
    node_state_df.at[i, "chk"] = 1

    def calc_next_node(i):
        for i_next_idx, link_cap in enumerate(current_network[i]):
            if link_cap > 0:
                tmp_dist = node_state_df.at[i, "dist"] + 1
                if tmp_dist < node_state_df.at[i_next_idx, "dist"]:
                    node_state_df.at[i_next_idx, "dist"] = tmp_dist
                    node_state_df.at[i_next_idx, "prev"] = i
        min_dist = float("inf")
        min_dist_node = None
        for i_next_idx, link_cap in enumerate(current_network[i]):
            if (
                node_state_df.at[i_next_idx, "chk"] == 0
                and node_state_df.at[i_next_idx, "dist"] < min_dist
            ):
                min_dist = node_state_df.at[i_next_idx, "dist"]
                min_dist_node = i_next_idx
        if min_dist_node == None:  # 経路が存在しない場合はNoneを返す
            return None
        node_state_df.at[min_dist_node, "chk"] = 1
        return min_dist_node

    while i != j and i != None:
        i = calc_next_node(i)

    if i == None:  # 経路が存在しない場合はNoneを返す
        return None

    route = []

    prev = j
    while prev != None:
        route.insert(0, prev)
        prev = node_state_df.at[prev, "prev"]

    return route


"""
最大路を返す

i,j : ノードのインデックス
network_shape : ネットワーク
"""


def routeMaxThp(i, j, current_network):
    def calc_next_node(start_node_idx, prev_idx):
        for i_next_idx, link_cap in enumerate(
            g_prime_network[start_node_idx]
        ):  # ノードjに到達
            if (i_next_idx not in prev_idx) and link_cap > 0 and i_next_idx == j:
                prev_idx.append(start_node_idx)
                prev_idx.append(j)
                return prev_idx
        prev_idx.append(start_node_idx)
        for i_next_idx, link_cap in enumerate(
            g_prime_network[start_node_idx]
        ):  # j以外のノードに進出
            if (i_next_idx not in prev_idx) and link_cap > 0:
                next_node = calc_next_node(i_next_idx, prev_idx)
                if next_node:  # ノード配列が返ってきたらそれを返す
                    return next_node
        return None  # 続くノードがなければNoneを返す

    # リンクを容量の大きい順にソートする
    link_cap_set = set()
    for row in current_network:
        for link_cap in row:
            link_cap_set.add(link_cap)

    # 一番大きい容量を持つリンクだけで構成されるグラフG'を生成する＝一番大きい容量を持つリンクだけでrouteTableを作る
    use_link_cap = set()  # グラフG'に追加されているリンクの容量

    route = None

    while route == None:
        # 一番大きい容量を取得
        max_cap = max(link_cap_set)
        if max_cap == 0:
            break
        link_cap_set.remove(max_cap)
        use_link_cap.add(max_cap)

        # グラフG'の作成
        g_prime_network = deepcopy(current_network)
        for row_idx, row in enumerate(g_prime_network):
            for col_idx, link_cap in enumerate(row):
                if link_cap not in use_link_cap:
                    g_prime_network[row_idx, col_idx] = 0

        # 3,グラフG'上でノードi->jの経路が存在するか調べる
        route = calc_next_node(i, [])

    # 存在しない場合は次に最大の重みを持つリンクをG'に追加する＝routeTableの更新
    return route


# ４つの経路選択方法で、それぞれn1~20で試し、結果を描画する

print("fixed min_hop")
for n in range(1, 21):
    print(n, make10kconnection("min_hop", "fixed", n, 10000, network_shape))
print("\n")

print("fixed max_thp")
for n in range(1, 21):
    print(n, make10kconnection("max_thp", "fixed", n, 10000, network_shape))
print("\n")

print("each min_hop")
for n in range(1, 21):
    print(n, make10kconnection("min_hop", "each", n, 10000, network_shape))
print("\n")

print("each max_thp")
for n in range(1, 21):
    print(n, make10kconnection("max_thp", "each", n, 10000, network_shape))
print("\n")
