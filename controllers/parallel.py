"""国家 AI 按国家并行规划（多进程）。

每个国家在回合开始时的同一份局面上各自规划（rule_ai.nation_turn），互不影响，所以可以分给多个进程
同时算；主进程再按国家编号依次重放各国的指令（rule_ai.merge）。单进程时走完全相同的流程（每国一份
局面副本），所以开不开多进程、开几个进程，结果都一模一样。

- 默认不开进程（脚本、测试里用 spawn 起进程会重新执行主脚本）；游戏服务器启动时调用 enable()。
- 进程数：环境变量 SEKAI_AI_WORKERS（0 = 不开进程）；默认 CPU 核数 − 1，最多 MAX_WORKERS。
- 只有 AI 国家够多、部队够多（MIN_JOBS / MIN_UNITS）才值得开进程：小局传数据的开销比算的还大。
- 工作进程按“路网版本”（GameState._path_epoch，道路 / 城市一变就换号）保留算过的距离场，下回合
  路网没变就接着用；地理信息（承载区等，只和地图有关）也按地图保留。
- 工作进程里 Numba 只用 1 个线程（多个进程已经占满核了）。
"""

from __future__ import annotations

import os
import pickle
import threading

MAX_WORKERS = 15
MIN_JOBS = 3  # 至少这么多个国家有 AI 才分进程
MIN_UNITS = 120  # 至少这么多支 AI 部队才分进程

_enabled = False
_pool = None
_pool_size = 0
_lock = threading.Lock()


def enable(on: bool = True) -> None:
    """允许开工作进程（服务器启动时调用；主模块要有 if __name__ == "__main__" 保护）。"""
    global _enabled
    _enabled = on


def workers() -> int:
    if not _enabled:
        return 0
    raw = os.environ.get("SEKAI_AI_WORKERS")
    if raw is not None:
        try:
            return max(0, min(MAX_WORKERS, int(raw)))
        except ValueError:
            return 0
    return max(0, min(MAX_WORKERS, (os.cpu_count() or 1) - 1))


def run(state, rules, jobs) -> list[dict]:
    """算各国的规划结果（按 jobs 的顺序返回）。"""
    n = workers()
    units = sum(_weight(j) for j in jobs)
    if n <= 1 or len(jobs) < MIN_JOBS or units < MIN_UNITS:
        return [plan(state, rules, job, share=True) for job in jobs]
    pool = _get_pool(n, rules)
    if pool is None:
        return [plan(state, rules, job, share=True) for job in jobs]
    # 按部队数均衡地分给各进程（大国先分，分给当前最轻的那组）；组内保持国家顺序
    groups = [[] for _ in range(min(n, len(jobs)))]
    load = [0] * len(groups)
    for job in sorted(jobs, key=lambda j: (-_weight(j), j["nation"])):
        k = min(range(len(groups)), key=lambda i: (load[i], i))
        groups[k].append(job)
        load[k] += 1 + _weight(job)
    payload = pickle.dumps(_slim(state), protocol=pickle.HIGHEST_PROTOCOL)
    futures = [pool.submit(_work, payload, group) for group in groups if group]
    results = {}
    try:
        for f in futures:
            for r in f.result():
                results[r["nation"]] = r
    except Exception:
        # 进程池坏了（被杀、内存不够……）：这回合在本进程算完，下回合重建进程池
        _reset_pool()
        return [plan(state, rules, job, share=True) for job in jobs]
    return [results[j["nation"]] for j in jobs]


def warm(rules) -> None:
    """先把工作进程拉起来、导入好模块（第一回合不用等进程启动）。"""
    n = workers()
    if n <= 1:
        return
    pool = _get_pool(n, rules)
    if pool is None:
        return
    try:
        for f in [pool.submit(_ping) for _ in range(n)]:
            f.result(timeout=300)
    except Exception:
        _reset_pool()


def _ping() -> bool:
    import controllers.rule_ai  # noqa: F401  导入引擎和 AI（含 Numba 内核）

    return True


def _weight(job) -> int:
    """估计这个国家要算多久（分组均衡用）：AI 看要决策的部队，城市托管看城市数（一城约顶 6 支部队）。"""
    if job.get("kind") == "cities":
        return 6 * job["weight"]
    return len(job["active"]) + len(job["units"]) // 4


def plan(state, rules, job, *, share: bool) -> dict:
    """在 state 的一份副本上算一个国家的规划。share：副本和 state 共用距离场缓存（单进程时）。"""
    from controllers import rule_ai
    from engine.events import EventLog
    from engine.rng import RngTree
    from engine.systems import encirclement
    from engine.systems.context import StepContext

    # 各国副本共用只读的派生数据：包围状态、领土索引（副本里单位一动、领土一变就各自作废重算）
    encirclement.status(state, rules)
    state.territory_index()
    local = state.clone()
    local._encircled = state._encircled
    local._territory_cache = state._territory_cache
    if share:
        # 路网没变时共用距离场（副本一改路网就换新字典，不影响别人）
        local._flow_cache = state._flow_cache
        local._edge_cache = state._edge_cache
    ctx = StepContext(
        state=local,
        rules=rules,
        orders=[],
        log=EventLog(turn=local.turn),
        rng=RngTree(local.seed, "turn", local.turn),
        record=[],
    )
    local._economy_cache_enabled = state._economy_cache_enabled
    nid = job["nation"]
    if job.get("kind") == "cities":
        from engine.systems import city_management

        city_management.nation_turn(ctx, nid)
    else:
        rule_ai.nation_turn(ctx, job)
    spans = ctx.auto_spans
    skip = set()
    for a, b in spans:
        skip.update(range(a, b))
    events = [e for i, e in enumerate(ctx.log.events) if i not in skip]
    nation = local.nations.get(nid)
    if job.get("kind") == "cities":
        return {
            "nation": nid,
            "orders": ctx.record,
            "events": events,
            "reports": {c.id: c.management for c in local.nation_cities(nid)},
            "management": nation.management,
            "road_plan": nation.road_plan,
            "road_failures": nation.road_failures,
        }
    return {
        "nation": nid,
        "orders": ctx.record,
        "routes": {
            uid: path
            for uid, path in ctx.planned_routes.items()
            if uid in local.units and local.units[uid].nation == nid
        },
        "strategy": nation.strategy if nation is not None else None,
        "notes": {
            uid: (local.units[uid].ai_goal, local.units[uid].ai_target)
            for uid in job["units"]
            if uid in local.units
        },
        "events": events,
    }


def _slim(state):
    """送进工作进程的局面：去掉只在本进程有用的缓存（距离场、通行费用表等由工作进程自己保留）。"""
    s = state.clone()
    s._flow_cache = {}
    s._edge_cache = {}
    s._settlement_cache = None
    s._settlement_usage = None
    s._worked_cache = {}
    s._yield_cache = {}
    s._facility_value_cache = {}
    s._supply_cache = {}
    s._power_cache = {}
    s._encircled = None
    s._territory_cache = None
    s._network_cache = None
    return s


# ---------------------------------------------------------------- 进程池


def _get_pool(n, rules):
    global _pool, _pool_size
    with _lock:
        if _pool is not None and _pool_size == n and _pool_rules is rules:
            return _pool
        _shutdown()
        try:
            import multiprocessing as mp
            from concurrent.futures import ProcessPoolExecutor

            ctx = mp.get_context("spawn")
            # 工作进程继承环境变量：Numba 只开 1 个线程（多个进程已经占满核了）
            os.environ["NUMBA_NUM_THREADS"] = "1"
            _pool = ProcessPoolExecutor(
                max_workers=n, mp_context=ctx, initializer=_init, initargs=(rules,)
            )
            _pool_size = n
            _set_rules(rules)
        except Exception:
            _pool = None
        return _pool


_pool_rules = None


def _set_rules(rules):
    global _pool_rules
    _pool_rules = rules


def _shutdown():
    global _pool
    if _pool is not None:
        _pool.shutdown(wait=False, cancel_futures=True)
        _pool = None


def _reset_pool():
    with _lock:
        _shutdown()


# ---------------------------------------------------------------- 工作进程

_RULES = None
_FLOWS: dict = {}  # (地图, 路网版本) → 距离场缓存
_GEO: dict = {}  # 地图 → 地理信息（承载区等）


def _init(rules):
    os.environ["NUMBA_NUM_THREADS"] = "1"
    global _RULES
    _RULES = rules


def _work(payload, jobs):
    state = pickle.loads(payload)
    g = state.grid
    world = (state.seed, g.width, g.height, tuple(sorted(g.tuning.items())), state.ruleset_id)
    key = (world, state._path_epoch)
    flows = _FLOWS.get(key)
    if flows is None:
        _FLOWS.clear()  # 路网变了：旧的距离场都作废
        flows = _FLOWS[key] = {}
    state._flow_cache = flows
    geo = _GEO.get(world)
    if geo is not None:
        state._settlement_cache = geo
    out = [plan(state, _RULES, job, share=True) for job in jobs]
    if state._settlement_cache is not None:
        if len(_GEO) > 4:
            _GEO.clear()
        _GEO[world] = state._settlement_cache
    return out
