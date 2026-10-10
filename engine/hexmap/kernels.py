"""地图生成里的热点循环，用 Numba 编译成机器码（cache=True：编译结果缓存在磁盘，只有第一次运行要编译）。

这里只放与业务无关的图算法：洪泛排水、连通分量、BFS 距离、多源 Dijkstra 生长、河流侵蚀、
河网汇流、邻居加权求和、Stommel 环流、造山传播、水汽迭代。所有函数只接受 NumPy 数组和标量，邻居表约定为 int32[N, 6]，-1 表示没有邻居。
"""

from __future__ import annotations

import heapq

import numpy as np
from numba import njit, prange


@njit(cache=True)
def priority_flood(elev: np.ndarray, is_water: np.ndarray, nb: np.ndarray) -> np.ndarray:
    """从所有水域往内陆洪泛：内陆格至少比流向它的格子高 1，消除积水洼地。"""
    n = elev.size
    out = elev.astype(np.int64).copy()
    visited = is_water.copy()
    heap = [(np.int64(0), np.int64(0))]
    heap.pop()
    for i in range(n):
        if visited[i]:
            heap.append((out[i], np.int64(i)))
    heapq.heapify(heap)
    while len(heap) > 0:
        e, cur = heapq.heappop(heap)
        for d in range(6):
            j = nb[cur, d]
            if j < 0 or visited[j]:
                continue
            visited[j] = True
            if out[j] <= e:
                out[j] = e + 1
            heapq.heappush(heap, (out[j], np.int64(j)))
    return out


@njit(cache=True)
def priority_flood_graph(
    height: np.ndarray, seeds: np.ndarray, adj: np.ndarray, jitter: np.ndarray
) -> np.ndarray:
    """任意图上的洪泛填洼（adj：int[N, k] 邻接表，-1 = 无）：从 seeds 往外，保证每个点都有
    一条严格下坡的路通向某个 seed。河网在"角"构成的图上走，格子上填过洼不代表角上没有洼点。

    填平的地方每步抬高 1 + jitter[j]（jitter ≥ 0）：洪泛前沿参差不齐，平地上的河道才会弯曲，
    而不是沿直线走。"""
    n = height.size
    out = height.astype(np.int64).copy()
    visited = seeds.copy()
    heap = [(np.int64(0), np.int64(0))]
    heap.pop()
    for i in range(n):
        if visited[i]:
            heap.append((out[i], np.int64(i)))
    heapq.heapify(heap)
    while len(heap) > 0:
        e, cur = heapq.heappop(heap)
        for d in range(adj.shape[1]):
            j = adj[cur, d]
            if j < 0 or visited[j]:
                continue
            visited[j] = True
            if out[j] <= e:
                out[j] = e + 1 + jitter[j]
            heapq.heappush(heap, (out[j], np.int64(j)))
    return out


@njit(cache=True)
def components(mask: np.ndarray, nb: np.ndarray) -> np.ndarray:
    """mask 为真的格子的连通分量编号（按最小 idx 的出现顺序），-1 = 不在 mask 内。"""
    n = mask.size
    labels = np.full(n, -1, dtype=np.int32)
    queue = np.empty(n, dtype=np.int64)
    label = 0
    for start in range(n):
        if not mask[start] or labels[start] >= 0:
            continue
        labels[start] = label
        head, tail = 0, 1
        queue[0] = start
        while head < tail:
            cur = queue[head]
            head += 1
            for d in range(6):
                j = nb[cur, d]
                if j >= 0 and mask[j] and labels[j] < 0:
                    labels[j] = label
                    queue[tail] = j
                    tail += 1
        label += 1
    return labels


@njit(cache=True)
def bfs_distance(sources: np.ndarray, nb: np.ndarray) -> np.ndarray:
    """到最近 source 格的步数；没有 source 时全为 2^20。"""
    n = sources.size
    dist = np.full(n, 1 << 20, dtype=np.int32)
    queue = np.empty(n, dtype=np.int64)
    tail = 0
    for i in range(n):
        if sources[i]:
            dist[i] = 0
            queue[tail] = i
            tail += 1
    head = 0
    while head < tail:
        cur = queue[head]
        head += 1
        for d in range(6):
            j = nb[cur, d]
            if j >= 0 and dist[j] > dist[cur] + 1:
                dist[j] = dist[cur] + 1
                queue[tail] = j
                tail += 1
    return dist


@njit(cache=True)
def masked_bfs(sources: np.ndarray, mask: np.ndarray, nb: np.ndarray, limit: int) -> np.ndarray:
    """只在 mask 为真的格子里走的 BFS 步数（比如只走海面），超过 limit 步就不再展开；
    走不到的格子为 2^20。"""
    n = sources.size
    dist = np.full(n, 1 << 20, dtype=np.int32)
    queue = np.empty(n, dtype=np.int64)
    tail = 0
    for i in range(n):
        if sources[i] and mask[i]:
            dist[i] = 0
            queue[tail] = i
            tail += 1
    head = 0
    while head < tail:
        cur = queue[head]
        head += 1
        if dist[cur] >= limit:
            continue
        for d in range(6):
            j = nb[cur, d]
            if j >= 0 and mask[j] and dist[j] > dist[cur] + 1:
                dist[j] = dist[cur] + 1
                queue[tail] = j
                tail += 1
    return dist


@njit(cache=True)
def grow_regions(
    seeds: np.ndarray,
    seed_rate: np.ndarray,
    tile_cost: np.ndarray,
    river_cost: float,
    river_edges: np.ndarray,
    in_mask: np.ndarray,
    nb: np.ndarray,
) -> np.ndarray:
    """多源 Dijkstra：各种子同时向外生长，每格归最先到达的种子。

    进入格子 j 的代价 = tile_cost[j] / seed_rate[种子]；跨过河流边另加 river_cost。
    """
    n = tile_cost.size
    owner = np.full(n, -1, dtype=np.int32)
    best = np.full(n, np.inf)
    heap = [(0.0, np.int64(0), np.int64(0))]
    heap.pop()
    for k in range(seeds.size):
        s = seeds[k]
        best[s] = 0.0
        heap.append((0.0, np.int64(k), np.int64(s)))
    heapq.heapify(heap)
    while len(heap) > 0:
        cost, label, cur = heapq.heappop(heap)
        if owner[cur] >= 0:
            continue
        owner[cur] = label
        for d in range(6):
            j = nb[cur, d]
            if j < 0 or not in_mask[j] or owner[j] >= 0:
                continue
            c = cost + tile_cost[j] / seed_rate[label]
            if (river_edges[cur] >> d) & 1:
                c += river_cost
            if c < best[j]:
                best[j] = c
                heapq.heappush(heap, (c, label, np.int64(j)))
    return owner


@njit(cache=True)
def stream_power(
    h: np.ndarray,
    is_water: np.ndarray,
    rain: np.ndarray,
    nb: np.ndarray,
    k: float,
    m: float,
    area_scale: float,
    iterations: int,
) -> np.ndarray:
    """河流侵蚀（Braun & Willett 隐式解法），见 hydrology.stream_power_erosion。"""
    n = h.size
    h = h.copy()
    receiver = np.full(n, -1, dtype=np.int64)
    acc = np.zeros(n)
    for _ in range(iterations):
        # 1. 下游：最低的邻居（水域按海平面 0 计）
        for i in range(n):
            receiver[i] = -1
            if is_water[i]:
                continue
            lowest = h[i]
            for d in range(6):
                j = nb[i, d]
                if j < 0:
                    continue
                hj = 0.0 if is_water[j] else h[j]
                if hj < lowest:
                    lowest = hj
                    receiver[i] = j
        # 2. 汇水面积：从高到低累加
        order = np.argsort(-h)
        for i in range(n):
            acc[i] = 0.0 if is_water[i] else rain[i] * area_scale
        for t in range(n):
            i = order[t]
            if receiver[i] >= 0:
                acc[receiver[i]] += acc[i]
        # 3. 隐式更新：从低到高
        for t in range(n - 1, -1, -1):
            i = order[t]
            r = receiver[i]
            if r < 0:
                continue
            base = 0.0 if is_water[r] else h[r]
            f = k * acc[i] ** m
            new = (h[i] + f * base) / (1.0 + f)
            h[i] = max(new, base + 0.5, 1.0)
    return h


@njit(cache=True, parallel=True)
def weighted_neighbor_sum(x: np.ndarray, nb: np.ndarray, w: np.ndarray) -> np.ndarray:
    """out[i] = Σ_d x[nb[i, d]] · w[i, d]（nb 里不应有 -1，调用方先换成自身）。"""
    n = x.size
    out = np.empty(n)
    for i in prange(n):
        s = 0.0
        for d in range(6):
            s += x[nb[i, d]] * w[i, d]
        out[i] = s
    return out


@njit(cache=True)
def _direction(nb: np.ndarray, p: int, q: int) -> int:
    for d in range(6):
        if nb[p, d] == q:
            return d
    return -1


@njit(cache=True)
def river_network(
    order: np.ndarray,
    height: np.ndarray,
    corner_key: np.ndarray,
    nxt: np.ndarray,
    touches_water: np.ndarray,
    runoff: np.ndarray,
    threshold: np.ndarray,
    headwater: float,
    min_len: int,
    edge_p: np.ndarray,
    edge_q: np.ndarray,
    hex_water: np.ndarray,
    nb: np.ndarray,
    river_edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """按汇流量生成河网（见 hydrology.trace_rivers），直接把河写进 river_edges。

    order：所有角按 (高度, 编号) 从高到低排列。每个角流向比它低的相邻角里最低的那个；
    从高到低累加径流得到流量，流量达到 threshold 处成河，成河之后一路向下游都是河；
    再从每个河源沿流量最大的上游角往上回溯（流量 ≥ headwater 为止），把河画到山里的源头。
    touches_water 只算入海口（湖泊不算：河流进湖再从出口流出，湖面上、湖岸上那段不写进 river_edges）。
    返回 (所有河流经过的角编号拼在一起, 每条河的起止偏移, 每个角的流量)。
    """
    k_count = height.size
    receiver = np.full(k_count, -1, dtype=np.int64)
    for c in range(k_count):
        if touches_water[c]:
            continue  # 入海 / 入湖口
        best = -1
        for e in range(3):
            k2 = nxt[c, e]
            if k2 < 0:
                continue
            lower = height[k2] < height[c] or (
                height[k2] == height[c] and corner_key[k2] < corner_key[c]
            )
            if not lower:
                continue
            if (
                best < 0
                or height[k2] < height[best]
                or (height[k2] == height[best] and corner_key[k2] < corner_key[best])
            ):
                best = k2
        receiver[c] = best

    # 流量：从高到低累加
    flow = runoff.copy()
    for t in range(k_count):
        c = order[t]
        r = receiver[c]
        if r >= 0:
            flow[r] += flow[c]

    # 能否流到水域（流进封闭洼地的不算河）：从低到高
    drains = touches_water.copy()
    for t in range(k_count - 1, -1, -1):
        c = order[t]
        r = receiver[c]
        if r >= 0 and drains[r]:
            drains[c] = True

    # 成河：流量够了，或者上游已经是河
    is_river = np.zeros(k_count, dtype=np.bool_)
    fed = np.zeros(k_count, dtype=np.bool_)  # 上游有河流进来
    for t in range(k_count):
        c = order[t]
        if touches_water[c] or not drains[c]:
            continue
        if fed[c] or flow[c] >= threshold[c]:
            is_river[c] = True
            r = receiver[c]
            if r >= 0:
                fed[r] = True

    # 回溯源头：沿流量最大的上游角往上走
    donor = np.full(k_count, -1, dtype=np.int64)
    for c in range(k_count):
        r = receiver[c]
        if r >= 0 and (donor[r] < 0 or flow[c] > flow[donor[r]]):
            donor[r] = c
    for t in range(k_count):
        c = order[t]
        if not is_river[c] or fed[c]:
            continue
        cur = c
        d = donor[cur]
        while d >= 0 and not is_river[d] and drains[d] and flow[d] >= headwater:
            is_river[d] = True
            fed[cur] = True
            cur = d
            d = donor[cur]

    # 拆成一条条河：从最高的源头开始一路走到入海口，后来的支流走到已有河道为止（汇流）
    taken = np.zeros(k_count, dtype=np.bool_)
    out = np.empty(2 * k_count + 1, dtype=np.int64)  # 每条河多存一个终点角
    offsets = np.zeros(k_count + 1, dtype=np.int64)
    pos = 0
    rivers = 0
    for t in range(k_count):
        s = order[t]
        if not is_river[s] or fed[s]:
            continue
        start = pos
        cur = s
        out[pos] = cur
        pos += 1
        joined = False
        while True:
            r = receiver[cur]
            out[pos] = r
            pos += 1
            if touches_water[r]:
                break
            if taken[r]:
                joined = True
                break
            cur = r
        length = pos - start - 1
        if length < min_len and not joined:
            pos = start  # 太短的沿海小河不要
            continue
        for i in range(start, pos - 1):
            taken[out[i]] = True
            a, b = out[i], out[i + 1]
            for e in range(3):
                if nxt[a, e] == b:
                    p, q = edge_p[a, e], edge_q[a, e]
                    if hex_water[p] or hex_water[q]:
                        continue  # 湖面上或湖岸线上
                    d = _direction(nb, p, q)
                    river_edges[p] |= np.uint8(1 << d)
                    river_edges[q] |= np.uint8(1 << ((d + 3) % 6))
        rivers += 1
        offsets[rivers] = pos
    return out[:pos], offsets[: rivers + 1], flow


@njit(cache=True)
def stommel_sor(
    forcing: np.ndarray,
    beta: np.ndarray,
    water: np.ndarray,
    r: float,
    dy: float,
    wrap: bool,
    iterations: int,
    omega: float,
) -> np.ndarray:
    """Stommel 风生环流：r·∇²ψ + β·∂ψ/∂X = curl τ，陆地与南北边界 ψ = 0，逐点超松弛迭代。

    数组都是 [行, 列]；X 向东（列，间距 1），Y 向北（行号增大是向南，间距 dy）。
    """
    h, w = forcing.shape
    psi = np.zeros((h, w))
    cx = r
    cy = r / (dy * dy)
    diag = 2.0 * cx + 2.0 * cy
    for _ in range(iterations):
        for i in range(h):
            for j in range(w):
                if not water[i, j]:
                    continue
                je = j + 1
                jw = j - 1
                if je >= w:
                    je = 0 if wrap else -1
                if jw < 0:
                    jw = w - 1 if wrap else -1
                pe = psi[i, je] if je >= 0 and water[i, je] else 0.0
                pw = psi[i, jw] if jw >= 0 and water[i, jw] else 0.0
                pn = psi[i - 1, j] if i > 0 and water[i - 1, j] else 0.0
                ps = psi[i + 1, j] if i < h - 1 and water[i + 1, j] else 0.0
                new = (
                    cx * (pe + pw) + cy * (pn + ps) + 0.5 * beta[i, j] * (pe - pw) - forcing[i, j]
                ) / diag
                psi[i, j] += omega * (new - psi[i, j])
    return psi


@njit(cache=True)
def bfs_nearest(
    sources: np.ndarray, allowed: np.ndarray, nb: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """多源 BFS：每格到最近 source 的步数，以及是哪个 source（只穿过 allowed 的格子）。

    到不了的格子：距离 2^20、来源 −1。
    """
    n = sources.size
    dist = np.full(n, 1 << 20, dtype=np.int32)
    origin = np.full(n, -1, dtype=np.int64)
    queue = np.empty(n, dtype=np.int64)
    tail = 0
    for i in range(n):
        if sources[i]:
            dist[i] = 0
            origin[i] = i
            queue[tail] = i
            tail += 1
    head = 0
    while head < tail:
        cur = queue[head]
        head += 1
        for d in range(6):
            j = nb[cur, d]
            if j >= 0 and allowed[j] and dist[j] > dist[cur] + 1:
                dist[j] = dist[cur] + 1
                origin[j] = origin[cur]
                queue[tail] = j
                tail += 1
    return dist, origin


@njit(cache=True)
def arrival_time(
    seeds: np.ndarray, speed: np.ndarray, step: np.ndarray, allowed: np.ndarray, nb: np.ndarray
) -> np.ndarray:
    """多源 Dijkstra 的到达时间：从各 seed 以各自的 speed 向外推进，进入格 j 用时
    step[cur, d] / speed[源]；只穿过 allowed 的格子。到不了的格子为 inf。

    用来算洋壳年龄：seed = 洋中脊，speed = 半扩张速率（km/百万年），step = 每步的公里数。
    """
    n = allowed.size
    best = np.full(n, np.inf)
    heap = [(0.0, np.int64(0), np.int64(0))]
    heap.pop()
    for k in range(seeds.size):
        s = seeds[k]
        best[s] = 0.0
        heap.append((0.0, np.int64(k), np.int64(s)))
    heapq.heapify(heap)
    while len(heap) > 0:
        t, k, cur = heapq.heappop(heap)
        if t > best[cur]:
            continue
        for d in range(6):
            j = nb[cur, d]
            if j < 0 or not allowed[j]:
                continue
            c = t + step[cur, d] / speed[k]
            if c < best[j]:
                best[j] = c
                heapq.heappush(heap, (c, k, np.int64(j)))
    return best


@njit(cache=True)
def _orogeny_value(p: float, w: float, o: float, d: float) -> float:
    ramp = 1.0
    if o > 0.0:
        ramp = min(max(d / o, 0.0), 1.0) ** 2
    x = max(d - o, 0.0) / max(w, 1e-9)
    return p * ramp / (1.0 + x**4)


@njit(cache=True)
def spread_orogeny(
    src: np.ndarray,
    peak: np.ndarray,
    width: np.ndarray,
    offset: np.ndarray,
    plate: np.ndarray,
    nb: np.ndarray,
    tile_km: float,
    passes: int,
) -> np.ndarray:
    """造山隆起向本板块内传播（见 topography._orogeny）：每格记住来源的峰值、宽度、偏移和距离，
    取造出最高隆起的那个来源。反复松弛 passes 遍。"""
    n = src.size
    dist = np.full(n, np.inf)
    sp = peak.copy()
    sw = width.copy()
    so = offset.copy()
    best = np.zeros(n)
    for i in range(n):
        if src[i]:
            dist[i] = 0.0
            best[i] = _orogeny_value(sp[i], sw[i], so[i], 0.0)
    for _ in range(passes):
        changed = False
        for i in range(n):
            for d in range(6):
                j = nb[i, d]
                if j < 0 or plate[j] != plate[i] or not np.isfinite(dist[j]):
                    continue
                cd = dist[j] + tile_km
                cv = _orogeny_value(sp[j], sw[j], so[j], cd)
                if cv > best[i]:
                    best[i] = cv
                    dist[i] = cd
                    sp[i] = sp[j]
                    sw[i] = sw[j]
                    so[i] = so[j]
                    changed = True
        if not changed:
            break
    return best


@njit(cache=True, parallel=True)
def moisture_loop(
    nb: np.ndarray,
    mix_w: np.ndarray,
    recycle: np.ndarray,
    water: np.ndarray,
    evap_rate: np.ndarray,
    sea_target: np.ndarray,
    rate: np.ndarray,
    threshold: np.ndarray,
    soft_width: np.ndarray,
    capacity: np.ndarray,
    iterations: int,
) -> np.ndarray:
    """水汽输送与降水的迭代（见 climate._moisture），返回后一半迭代的平均降水。"""
    n = water.size
    humidity = np.zeros(n)
    nxt = np.zeros(n)
    soil = np.zeros(n)
    total = np.zeros(n)
    start = iterations // 2
    for step in range(iterations):
        for i in prange(n):
            h = recycle[i] * soil[i]
            for d in range(6):
                h += humidity[nb[i, d]] * mix_w[i, d]
            if water[i]:
                e = evap_rate[i] * (sea_target[i] - h)
                if e > 0.0:
                    h += e
            # 平滑的 max(h − 阈值, 0)
            z = (h - threshold[i]) / max(soft_width[i], 1e-9)
            sp = z if z > 30.0 else np.log1p(np.exp(z))
            rain = rate[i] * soft_width[i] * sp
            h -= rain
            if h > capacity[i]:
                rain += h - capacity[i]
                h = capacity[i]
            soil[i] = 0.85 * soil[i] + 0.15 * rain
            nxt[i] = h
            if step >= start:
                total[i] += rain
        for i in prange(n):
            humidity[i] = nxt[i]
    return total / (iterations - start)


# ---------------------------------------------------------------- AI 寻路（和 Python 版逐步一致，结果相同）
# nogil=True：运行时释放 GIL，服务器里几局游戏可以在不同线程里同时算（多核）。
FIELD_INF = 1 << 40


@njit(cache=True, nogil=True)
def flow_field(
    edges: np.ndarray,
    directions: np.ndarray,
    stand: np.ndarray,
    blocked: np.ndarray,
    sources: np.ndarray,
    limit: int,
) -> np.ndarray:
    """以 sources 为终点的反向 Dijkstra（共享距离场）：dist[t] = 从 t 走到终点的最小代价，走不到为 FIELD_INF。
    edges[n, d]：从 n 往方向 d 走一步的代价（0 = 不能走）；从 t 进入邻居 n（t 在 n 的 (d+3)%6 方向）
    用 edges[n, (d + 3) % 6]。最多展开 limit 个节点。和 controllers/rule_ai.py 的旧 Python 实现一致。"""
    n = edges.shape[0]
    dist = np.full(n, FIELD_INF, dtype=np.int64)
    heap = [(np.int64(0), np.int64(0))]
    heap.pop()
    for t in sources:
        dist[t] = 0
        heap.append((np.int64(0), np.int64(t)))
    heapq.heapify(heap)
    expanded = 0
    while len(heap) > 0 and expanded < limit:
        value, tile = heapq.heappop(heap)
        if value != dist[tile]:
            continue
        expanded += 1
        for d in range(6):
            nb = directions[tile, d]
            if nb < 0 or blocked[nb]:
                continue
            cost = edges[nb, (d + 3) % 6]
            if cost <= 0 or not stand[nb]:
                continue
            new = value + cost
            if new < dist[nb]:
                dist[nb] = new
                heapq.heappush(heap, (new, np.int64(nb)))
    return dist


@njit(cache=True, nogil=True, parallel=True)
def flow_fields(
    edges: np.ndarray,
    directions: np.ndarray,
    stand: np.ndarray,
    blocked: np.ndarray,
    sources: np.ndarray,
    starts: np.ndarray,
    limit: int,
) -> np.ndarray:
    """一次算好几张距离场（每张的终点是 sources[starts[k]:starts[k+1]]），多核并行；每张和 flow_field 相同。"""
    k = starts.size - 1
    out = np.empty((k, edges.shape[0]), dtype=np.int64)
    for i in prange(k):
        out[i] = flow_field(
            edges, directions, stand, blocked, sources[starts[i] : starts[i + 1]], limit
        )
    return out


@njit(cache=True, nogil=True)
def astar(
    edges: np.ndarray,
    directions: np.ndarray,
    blocked: np.ndarray,
    start: int,
    goal: int,
    max_expand: int,
    q: np.ndarray,
    r: np.ndarray,
    width: int,
    wrap: bool,
) -> np.ndarray:
    """start → goal 的最省路径（含两端），走不到返回空数组。edges[a, d]：从 a 往方向 d 走一步的代价。
    和 engine/hexmap/pathfind.py 的 find_path（edge_cost 版）逐步一致。"""
    empty = np.empty(0, dtype=np.int64)
    if start == goal:
        out = np.empty(1, dtype=np.int64)
        out[0] = start
        return out
    if blocked[goal]:
        return empty
    n = edges.shape[0]
    gq, gr = q[goal], r[goal]
    best = np.full(n, FIELD_INF, dtype=np.int64)
    parent = np.full(n, -1, dtype=np.int64)
    best[start] = 0

    def h(i):
        dq, dr = q[i] - gq, r[i] - gr
        b = (abs(dq) + abs(dr) + abs(dq + dr)) // 2
        if wrap:
            for k in (-1, 1):
                dqk = dq - k * width
                v = (abs(dqk) + abs(dr) + abs(dqk + dr)) // 2
                if v < b:
                    b = v
        return b

    heap = [(np.int64(h(start)), np.int64(0), np.int64(start))]
    expanded = 0
    while len(heap) > 0:
        _, gg, cur = heapq.heappop(heap)
        if cur == goal:
            length = 1
            t = cur
            while parent[t] >= 0:
                t = parent[t]
                length += 1
            out = np.empty(length, dtype=np.int64)
            t = cur
            for i in range(length - 1, -1, -1):
                out[i] = t
                t = parent[t]
            return out
        if gg > best[cur]:
            continue
        expanded += 1
        if expanded > max_expand:
            return empty
        for d in range(6):
            nb = directions[cur, d]
            if nb < 0 or blocked[nb]:
                continue
            c = edges[cur, d]
            if c <= 0:
                continue
            ng = gg + c
            if ng < best[nb]:
                best[nb] = ng
                parent[nb] = cur
                heapq.heappush(heap, (np.int64(ng + h(nb)), np.int64(ng), np.int64(nb)))
    return empty
