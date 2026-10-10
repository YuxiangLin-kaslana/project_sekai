"""Web 服务：世界生成与地图浏览、对局（建对局、看快照、提交指令结算一回合）。WebSocket 以后也挂在这里。

uv run uvicorn server.app:app --host 127.0.0.1 --port 8095
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ValidationError

from engine.actions import parse_orders
from engine.hexmap.worldgen import WorldGenParams, generate_world
from engine.rules.loader import load_ruleset
from engine.setup import new_game
from engine.state import MOVE_STEPS
from engine.step import step
from server.game import GameStore, advance, rules_json, snapshot
from server.mapdata import MapLayers, build_layers, chunk_bytes

STATIC = Path(__file__).parent / "static"
RULES = load_ruleset("alt_ww2")
CACHE_SIZE = 6
CACHE_TILES = 1_200_000  # 缓存里所有地图的格子总数上限（约 3 张 800×460 精细地球）
LONG_CACHE = {"Cache-Control": "public, max-age=86400"}


def _data_version() -> str:
    """生成地图数据的代码 + 规则集的指纹。

    世界 / 分块接口会被浏览器和 Cloudflare 缓存一天；地址里带上这个指纹，
    改了生成算法或数据格式并重启后，旧缓存就不会再被拿来配新页面。
    """
    root = Path(__file__).parent.parent
    h = hashlib.blake2b(digest_size=6)
    files = [*sorted((root / "engine").rglob("*.py")), root / "server" / "mapdata.py"]
    files += sorted((root / "rules").rglob("*.yaml"))
    for f in files:
        h.update(f.relative_to(root).as_posix().encode())
        h.update(f.read_bytes())
    return h.hexdigest()


DATA_VERSION = _data_version()


def _warmup() -> None:
    """后台预热：生成一张小图、让 AI 跑两回合，把地图生成、AI 距离场、A* 寻路的 Numba 内核都编译好
    （Windows 版第一次启动要编译一两分钟；不预热的话第一回合会卡住）。"""
    try:
        state = new_game(generate_world(WorldGenParams(width=40, height=25, seed=0), RULES), RULES)
        orders = parse_orders(
            [{"type": "set_ai", "issuer": f"nation:{n}", "enabled": True} for n in state.nations]
        )
        state = step(state, orders, RULES, immediate=True).state
        for _ in range(2):
            state = step(state, [], RULES).state
    except Exception:  # noqa: BLE001 —— 预热失败不影响正常使用
        pass


@asynccontextmanager
async def _lifespan(_: FastAPI):
    # 地图生成的热点用 Numba 编译；启动后在后台先生成一张小图，把编译做掉，第一个真实请求就不用等
    # 国家 AI 按国家并行规划：服务器里才开工作进程（见 controllers/parallel.py），启动时先把进程拉起来
    from controllers import parallel

    parallel.enable()
    threading.Thread(target=parallel.warm, args=(RULES,), daemon=True).start()
    threading.Thread(target=_warmup, daemon=True).start()
    yield


app = FastAPI(
    title="Project Sekai",
    docs_url="/api/docs",
    openapi_url="/api/openapi.json",
    lifespan=_lifespan,
)
# 地图分块是 application/octet-stream，Caddy 和 Cloudflare 都不压缩；gzip 后约小 60%（精细地球全图 7 MB → 3 MB）
app.add_middleware(GZipMiddleware, minimum_size=1024, compresslevel=5)
app.mount("/static", StaticFiles(directory=STATIC), name="static")
# Windows 下载包（tools/windows/build.py 打到 downloads/windows/：完整包、更新包、latest.json、下载页）
app.mount(
    "/downloads",
    StaticFiles(
        directory=Path(__file__).resolve().parents[1] / "downloads", html=True, check_dir=False
    ),
    name="downloads",
)

Seed = Annotated[int, Query(ge=0, le=2_147_483_647)]
Width = Annotated[int, Query(ge=20, le=800)]
Height = Annotated[int, Query(ge=16, le=460)]
Nations = Annotated[int, Query(ge=1, le=40)]
Land = Annotated[float | None, Query(ge=0.1, le=0.9)]
Temp = Annotated[float, Query(ge=-1.0, le=1.0)]
Rain = Annotated[float, Query(ge=-1.0, le=1.0)]
Rough = Annotated[float, Query(ge=0.0, le=1.0)]

_cache: OrderedDict[tuple, MapLayers] = OrderedDict()
_lock = threading.Lock()


def _layers(
    seed: int,
    width: int,
    height: int,
    nations: int,
    land: float | None,
    temp: float,
    rain: float,
    rough: float,
) -> MapLayers:
    """同参数只生成一次（最近 CACHE_SIZE 张，格子总数不超过 CACHE_TILES）；生成过程串行，避免同一张图被并发重复生成。"""
    key = (
        seed,
        width,
        height,
        nations,
        None if land is None else round(land, 3),
        round(temp, 2),
        round(rain, 2),
        round(rough, 2),
    )
    with _lock:
        if key in _cache:
            _cache.move_to_end(key)
            return _cache[key]
        params = WorldGenParams(
            width=width,
            height=height,
            seed=seed,
            num_starts=nations,
            land_ratio=key[4],
            temperature=key[5],
            rainfall=key[6],
            roughness=key[7],
        )
        layers = build_layers(generate_world(params, RULES), RULES)
        _cache[key] = layers

        def tiles() -> int:
            return sum(k[1] * k[2] for k in _cache)

        while len(_cache) > 1 and (len(_cache) > CACHE_SIZE or tiles() > CACHE_TILES):
            _cache.popitem(last=False)
        return layers


@app.get("/api/world")
def world_meta(
    seed: Seed = 42,
    width: Width = 80,
    height: Height = 50,
    nations: Nations = 8,
    land: Land = None,
    temp: Temp = 0.0,
    rain: Rain = 0.0,
    rough: Rough = 0.5,
) -> Response:
    layers = _layers(seed, width, height, nations, land, temp, rain, rough)
    body = json.dumps(layers.meta, ensure_ascii=False, separators=(",", ":"))
    return Response(body, media_type="application/json", headers=LONG_CACHE)


@app.get("/api/world/chunk")
def world_chunk(
    cx: Annotated[int, Query(ge=0)],
    cy: Annotated[int, Query(ge=0)],
    seed: Seed = 42,
    width: Width = 80,
    height: Height = 50,
    nations: Nations = 8,
    land: Land = None,
    temp: Temp = 0.0,
    rain: Rain = 0.0,
    rough: Rough = 0.5,
) -> Response:
    layers = _layers(seed, width, height, nations, land, temp, rain, rough)
    try:
        data = chunk_bytes(layers, cx, cy)
    except IndexError as e:
        raise HTTPException(404, str(e)) from e
    return Response(data, media_type="application/octet-stream", headers=LONG_CACHE)


# ---------------------------------------------------------------- 对局

GAMES = GameStore()


class GameCreate(BaseModel):
    """建对局：在这张世界（参数同 /api/world）上按起始点摆放国家。"""

    seed: int = Field(42, ge=0, le=2_147_483_647)
    width: int = Field(80, ge=20, le=800)
    height: int = Field(50, ge=16, le=460)
    nations: int = Field(8, ge=1, le=40)
    land: float | None = Field(None, ge=0.1, le=0.9)
    temp: float = Field(0.0, ge=-1.0, le=1.0)
    rain: float = Field(0.0, ge=-1.0, le=1.0)
    rough: float = Field(0.5, ge=0.0, le=1.0)
    stress_mode: bool = False
    # 开局设置（百分比，100 = 默认），见 engine/hexmap/grid.py 的 HexGrid.tuning
    land_speed: int = Field(100, ge=50, le=300)
    sea_speed: int = Field(100, ge=50, le=300)
    city_size: int = Field(100, ge=50, le=200)
    start_gold: int = Field(100, ge=0, le=1000)
    upkeep: int = Field(100, ge=0, le=300)
    air_range: int = Field(100, ge=50, le=300)
    naval_range: int = Field(100, ge=50, le=300)
    artillery_range: int = Field(100, ge=50, le=300)
    capacity: int = Field(100, ge=50, le=400)
    ai_budget: int = Field(128, ge=64, le=256)  # 每国每回合最多决策几支部队（不是百分比）


TUNING_KEYS = (
    "land_speed",
    "sea_speed",
    "city_size",
    "start_gold",
    "upkeep",
    "air_range",
    "naval_range",
    "artillery_range",
    "capacity",
)


class TurnSubmit(BaseModel):
    orders: list[dict] = Field(default_factory=list, max_length=2000)
    autoplay: bool = True
    expected_hash: str | None = None


def _game(game_id: str):
    game = GAMES.get(game_id)
    if game is None:
        raise HTTPException(404, "对局不存在（服务重启后内存里的对局会丢失）")
    return game


@app.post("/api/games")
def create_game(body: GameCreate) -> dict:
    p = body.model_dump()
    dense = p.pop("stress_mode")
    tuning = {k: v for k in TUNING_KEYS if (v := p.pop(k)) != 100}
    if (budget := p.pop("ai_budget")) != 128:
        tuning["ai_budget"] = budget
    layers = _layers(**p)
    if dense:
        from engine.stress import populate

        started = time.perf_counter()
        state, stats = populate(layers.world, RULES, tuning=tuning)
        stats["generation_ms"] = round((time.perf_counter() - started) * 1000, 1)
        game = GAMES.create(p, state, stats)
    else:
        game = GAMES.create(p, new_game(layers.world, RULES, tuning))
    return {"rules": rules_json(RULES), **snapshot(game, RULES)}


@app.get("/api/games/{game_id}")
def get_game(game_id: str) -> dict:
    game = _game(game_id)
    with game.lock:
        return {"rules": rules_json(RULES), **snapshot(game, RULES)}


@app.post("/api/games/{game_id}/turn")
def play_turn(game_id: str, body: TurnSubmit) -> dict:
    """提交本回合指令并结算（目前所有国家的指令一起提交，同时回合制的调度以后做）。"""
    game = _game(game_id)
    try:
        orders = parse_orders(body.orders)
    except ValidationError as e:
        raise HTTPException(422, e.errors(include_url=False)) from e
    with game.lock:
        result = advance(game, orders, RULES, autoplay=body.autoplay)
        return {**snapshot(game, RULES), "rejected": result["rejected"]}


@app.post("/api/games/{game_id}/benchmark")
def benchmark_game(game_id: str) -> dict:
    """Measure a real turn on step's private clone, preserving the player's game."""
    game = _game(game_id)
    with game.lock:
        cpu = time.process_time()
        started = time.perf_counter()
        result = step(game.state, [], RULES)
        elapsed = time.perf_counter() - started
        out = {
            "turn_ms": round(elapsed * 1000, 1),
            "cpu_ms": round((time.process_time() - cpu) * 1000, 1),
            "simulated_turn": result.state.turn,
            "events": len(result.events),
        }
        try:  # 峰值内存只有 Linux / macOS 拿得到（Windows 版没有 resource 模块）
            import resource

            out["server_peak_rss_mb"] = round(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1
            )
        except ImportError:
            pass
        return out


@app.post("/api/games/{game_id}/actions")
def play_actions(game_id: str, body: TurnSubmit) -> dict:
    """Execute commands in the current turn without refreshing movement or production."""
    game = _game(game_id)
    try:
        orders = parse_orders(body.orders)
    except ValidationError as e:
        raise HTTPException(422, e.errors(include_url=False)) from e
    with game.lock:
        if body.expected_hash is not None and body.expected_hash != game.state.state_hash():
            raise HTTPException(409, "对局已经变化，请刷新后重新选择")
        result = advance(game, orders, RULES, autoplay=False, immediate=True)
        return {**snapshot(game, RULES), "rejected": result["rejected"]}


@app.get("/api/games/{game_id}/tactics")
def game_tactics(game_id: str, actor: str, col: int | None = None, row: int | None = None) -> dict:
    from engine.ids import parse_ref, ref
    from engine.systems.combat import attack_kind, strength, tactics
    from engine.systems.movement import group_members, has_room, movement_map, route_for

    game = _game(game_id)
    with game.lock:
        s = game.state
        kind = "city" if actor.startswith("city:") else "unit"
        try:
            entity = (s.cities if kind == "city" else s.units)[parse_ref(actor, kind)]
        except (ValueError, KeyError) as e:
            raise HTTPException(404, "单位或城市不存在") from e
        result = {
            "actor": actor,
            "state_hash": s.state_hash(),
            **tactics(s, RULES, entity),
            "combat_strength": strength(s, RULES, entity),
            "ranged_strength": strength(s, RULES, entity, ranged=True)
            if attack_kind(RULES, entity) != "melee"
            else None,
        }
        if kind == "unit":
            from engine.systems.pillage import pillage_target

            target, problem = pillage_target(s, RULES, entity)
            result["pillage"] = {"name": target["name"], "loot": target["loot"]} if target else None
            result["pillage_blocked"] = problem
            from engine.systems import roads as road_rules

            if "build_facility" in RULES.units[entity.type].tags:
                # 工兵修路：往哪几个相邻格能修、各要多少金（方向按地图上的方位）
                import math

                g = s.grid
                lay = []
                for b in g.neighbors_of(entity.tile):
                    if road_rules.lay_problem(s, RULES, entity, b) is None:
                        kind, price = road_rules.engineering(
                            s, RULES, entity.nation, entity.tile, b
                        )
                        dq = int(g.axial_q[b]) - int(g.axial_q[entity.tile])
                        dr = int(g.axial_r[b]) - int(g.axial_r[entity.tile])
                        if g.wrap_x and abs(dq) > 1:
                            dq = -1 if dq > 0 else 1
                        angle = math.degrees(math.atan2(-dr * 0.866, dq + dr / 2)) % 360
                        names = ["东", "东北", "北", "西北", "西", "西南", "南", "东南"]
                        lay.append(
                            {
                                "to": list(g.col_row(b)),
                                "cost": price,
                                "kind": kind,
                                "dir": names[round(angle / 45) % 8],
                            }
                        )
                result["lay_road"] = lay
            result["demolish_blocked"] = road_rules.demolish_problem(s, RULES, entity)
            from engine.systems.naval import embark_target

            if RULES.units[entity.type].domain == "land":
                ship, problem = embark_target(s, RULES, entity)
                result["embark"] = RULES.units[ship.type].name if ship else None
                result["embark_blocked"] = problem
            from engine.systems import generals

            if generals.is_general(RULES, entity):
                # 将领：军团成员、上限、加成，以及身边还能编入的本国军事单位
                corps = generals.members(s, entity)
                room = generals.capacity(RULES, entity.rank) - len(corps)
                nearby = [
                    u
                    for t in s.grid.within(entity.tile, RULES.generals.radius)
                    for u in s.units_at(t)
                    if u.nation == entity.nation
                    and u.corps != entity.id
                    and RULES.units[u.type].category == "military"
                ]
                result["corps"] = {
                    "members": [ref("unit", u.id) for u in corps],
                    "capacity": generals.capacity(RULES, entity.rank),
                    "bonus": generals.bonus(RULES, entity.rank),
                    "radius": RULES.generals.radius,
                    "recruits": [ref("unit", u.id) for u in sorted(nearby, key=lambda u: u.id)][
                        : max(0, room)
                    ],
                }
            elif entity.corps is not None and entity.corps in s.units:
                g = s.units[entity.corps]
                result["commander"] = {
                    "general": ref("unit", g.id),
                    "name": g.name,
                    "rank": g.rank,
                    "title": g.title,
                    "in_range": generals.commander(s, RULES, entity) is not None,
                    "bonus": generals.bonus(RULES, g.rank),
                }
            if "found_city" in RULES.units[entity.type].tags:
                from engine.systems.cities import found_city_problem

                result["found_city_blocked"] = found_city_problem(
                    s, RULES, entity.nation, entity.tile
                )
            best, parents = movement_map(s, RULES, entity)
            paths = {}
            for tile in sorted(best):
                if tile == entity.tile or not has_room(s, RULES, entity, tile):
                    continue  # 路过的已满格子不能停
                path = [tile]
                while path[-1] != entity.tile:
                    path.append(parents[path[-1]])
                paths[str(tile)] = path[::-1]
            result.update(
                move_tiles=sorted(paths, key=int),
                move_paths=paths,
                movement_left=min(
                    u.moves_left + u.road_credit / MOVE_STEPS for u in group_members(s, entity)
                ),
            )
            result["move_tiles"] = [int(t) for t in result["move_tiles"]]
            if col is not None and row is not None:
                try:
                    tile = s.grid.idx(col, row)
                except IndexError as e:
                    raise HTTPException(422, str(e)) from e
                result["route"] = route_for(s, RULES, entity, tile)
        return result


@app.get("/healthz")
def healthz() -> dict:
    return {"ok": True, "ruleset": RULES.id, "data": DATA_VERSION}


_update = {"checked": 0.0, "manifest": None}
UPDATE_EVERY = 2 * 3600  # Windows 版运行中每两小时查一次更新


def _version_tuple(text: str) -> tuple[int, ...]:
    try:
        return tuple(int(x) for x in str(text).strip().split("."))
    except ValueError:
        return (0,)


@app.get("/api/client/update")
def client_update() -> dict:
    """Windows 版（启动器设了 SEKAI_DESKTOP）：有没有新版本。网页版只回当前版本。
    新版本运行环境没变时，关掉游戏重新启动就自动更新；变了要重新下完整包（full_required）。"""
    current = os.environ.get("SEKAI_VERSION", "")
    if os.environ.get("SEKAI_DESKTOP") != "1":
        return {"desktop": False, "current": current}
    if time.time() - _update["checked"] > UPDATE_EVERY:
        _update["checked"] = time.time()
        try:
            import urllib.request

            url = os.environ.get("SEKAI_UPDATE_URL", "")
            req = urllib.request.Request(url, headers={"User-Agent": "SekaiDesktop"})
            with urllib.request.urlopen(req, timeout=8) as r:
                _update["manifest"] = json.loads(r.read().decode("utf-8"))
        except Exception:  # noqa: BLE001 —— 离线时不提示
            pass
    m = _update["manifest"] or {}
    latest = m.get("version", "")
    newer = bool(latest) and _version_tuple(latest) > _version_tuple(current)
    return {
        "desktop": True,
        "current": current,
        "latest": latest,
        "available": newer,
        "full_required": newer and m.get("runtime") != os.environ.get("SEKAI_RUNTIME"),
        "page_url": m.get("page_url"),
    }


class RoadPreview(BaseModel):
    issuer: str
    start: tuple[int, int]
    to: tuple[int, int]


@app.get("/api/games/{game_id}/city-options")
def city_options(game_id: str, city: str) -> dict:
    from engine.ids import parse_ref
    from engine.systems.cities import city_tiles, facility_options

    game = _game(game_id)
    try:
        cid = parse_ref(city, "city")
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    with game.lock:
        c = game.state.cities.get(cid)
        if c is None:
            raise HTTPException(404, "没有这座城市")
        options = []
        if c.construction is None:
            for tile in city_tiles(game.state, c):
                if any(u.task for u in game.state.units_at(tile)):
                    continue
                buildings = facility_options(game.state, RULES, c.nation, tile)
                if buildings:
                    options.append(
                        {
                            "tile": tile,
                            "to": game.state.grid.col_row(tile),
                            "facilities": buildings,
                        }
                    )
        return {"state_hash": game.state.state_hash(), "city": city, "options": options}


@app.post("/api/games/{game_id}/road-plan")
def road_preview(game_id: str, body: RoadPreview) -> dict:
    from engine.ids import parse_ref
    from engine.systems.roads import plan

    game = _game(game_id)
    try:
        nation = parse_ref(body.issuer, "nation")
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    with game.lock:
        return {
            "state_hash": game.state.state_hash(),
            **plan(game.state, RULES, nation, body.start, body.to),
        }


def _asset_version(name: str) -> str:
    st = (STATIC / name).stat()
    return f"{int(st.st_mtime)}-{st.st_size}"


@app.get("/", include_in_schema=False)
def index() -> HTMLResponse:
    # 脚本地址带上版本号：代码一更新地址就变，浏览器和 Cloudflare 不会拿旧缓存配新页面；
    # d = 数据版本，viewer 会把它带进所有 /api/world 请求
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    for name in ("viewer.js", "game.js"):
        version = f"v={_asset_version(name)}&d={DATA_VERSION}"
        html = html.replace(f'"/static/{name}"', f'"/static/{name}?{version}"')
    return HTMLResponse(html, headers={"Cache-Control": "no-cache"})
