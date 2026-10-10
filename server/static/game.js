// Project Sekai 对局图层：国家领土、城市、城外设施、单位牌（兵牌）
// - 数据来自 /api/games 的快照；不依赖 viewer.js，由页面通过 viewer.setOverlay 挂上去
// - 按缩放分层（架构设计 11.3）：远 = 领土色块 + 城市点；中 = 城市名牌 + 单位牌堆（顶牌 + 数量）；
//   近 = 完整单位牌（战斗力、兵种符号、数量、血条）+ 设施图标
// - 单位牌：底色 = 国家色，中间是北约军标风格的兵种符号；同格多张错开叠放，最上面是战斗力最高的；
//   航空单位用圆角菱形，悬浮在地表牌右上方并带投影

const SQRT3 = Math.sqrt(3);
const CORNERS = [0, 1, 2, 3, 4, 5].map((i) => {
  const a = ((60 * i - 30) * Math.PI) / 180;
  return [Math.cos(a), Math.sin(a)];
});
const EDGE_CORNERS = [[0, 1], [5, 0], [4, 5], [3, 4], [2, 3], [1, 2]]; // 方向 E NE NW W SW SE 的边
const DIRS = [[1, 0], [1, -1], [0, -1], [-1, 0], [-1, 1], [0, 1]]; // 轴向坐标方向
const FONT = '-apple-system, "PingFang SC", "Noto Sans SC", "Microsoft YaHei", sans-serif';

// 缩放阈值（格子半径，CSS 像素）
const LOD = { cityPlate: 3, armyDot: 3, counters: 7, fullCard: 30, tiles: 3, tileArt: 6 };

export const G = {
  rules: null, // { units, buildings, facilities, stacking }
  snap: null,
  W: 0,
  H: 0,
  nations: new Map(), // "nation:1" → { ...nation, idx, rgb }
  cities: new Map(), // "city:9" → city
  units: new Map(), // "unit:12" → unit
  ownerOf: null, // Int16Array：每格所属国家序号（-1 无主）
  cityAt: new Map(), // 市中心格 → city
  stacks: new Map(), // 格子 → { surface: [unit...], air: [unit...] }（已排序，第一个是顶牌）
  facilityAt: new Map(), // 格子 → 设施规则 ID
  owned: [], // [格子, 国家序号]
  selected: null, // { tile }：选中查看的格子
  layers: { territory: true, cities: true, units: true, facilities: true, plans: true },
  political: false, // "国家"模式：底图不画地形，领土实心铺色，远景标国名
  cityOf: null, // Int32Array：每格属于第几座城（-1 无），画同一国家内城市辖区的分界
  labels: [], // 每个国家的国名位置：{ name, rgb, x, y, size }（世界坐标）
};

const POLITICAL_LABELS_BELOW = 6; // "国家"模式下缩放小于它时标国名、不画城市名牌
const TERRAIN_MUTE = 0.3; // 有对局时地形底图去掉多少饱和度（0 = 不变，1 = 全灰）

function hexRgb(c) {
  if (!c.startsWith("#")) return (c.match(/[\d.]+/g) || [120, 120, 120]).slice(0, 3).map(Number);
  const n = parseInt(c.slice(1), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}
function rgba([r, g, b], a) {
  return `rgba(${r},${g},${b},${a})`;
}
function shade([r, g, b], f) {
  // f < 0 变暗，> 0 变亮
  const t = f < 0 ? 0 : 255;
  const k = Math.abs(f);
  return `rgb(${Math.round(r + (t - r) * k)},${Math.round(g + (t - g) * k)},${Math.round(b + (t - b) * k)})`;
}

// ---------------------------------------------------------------- 数据

function colRow(tile) {
  return [tile % G.W, (tile / G.W) | 0];
}
function center(tile) {
  const [col, row] = colRow(tile);
  return [SQRT3 * (col + 0.5 * (row & 1)), 1.5 * row];
}
function neighbor(tile, d) {
  const [col, row] = colRow(tile);
  const q = col - (row - (row & 1)) / 2;
  const nq = q + DIRS[d][0];
  const nr = row + DIRS[d][1];
  if (nr < 0 || nr >= G.H) return -1;
  const ncol = (((nq + (nr - (nr & 1)) / 2) % G.W) + G.W) % G.W;
  return nr * G.W + ncol;
}

export function unitDef(u) {
  return G.rules.units[u.type];
}

// 顶牌顺序：军事单位按战斗力从高到低，平民在后；同等按 ID
function stackOrder(a, b) {
  const da = unitDef(a), db = unitDef(b);
  const ma = da.category === "military" ? 1 : 0, mb = db.category === "military" ? 1 : 0;
  if (ma !== mb) return mb - ma;
  const sa = Math.max(da.strength, da.ranged_strength || da.anti_air || 0), sb = Math.max(db.strength, db.ranged_strength || db.anti_air || 0);
  if (sa !== sb) return sb - sa;
  return parseInt(a.id.slice(5), 10) - parseInt(b.id.slice(5), 10);
}

// ---- 快照解码：服务器只发原始数组（见 server/game.py 的 pack / pack_roads），坐标、等级、
// 每格归属和道路都在这里展开

// pack() 的逆运算：游程编码 {rle: [值, 长度, ...]} 或 base64 小端原始字节 {b64, dtype}
export function unpack(p) {
  if (p.rle) {
    const out = new Int32Array(p.n);
    for (let i = 0, k = 0; i < p.rle.length; i += 2) {
      out.fill(p.rle[i], k, k + p.rle[i + 1]);
      k += p.rle[i + 1];
    }
    return out;
  }
  const bin = atob(p.b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  const T = { uint8: Uint8Array, uint16: Uint16Array, int16: Int16Array, int32: Int32Array }[p.dtype];
  return new T(bytes.buffer);
}

// 经验 → 等级（与 engine/systems/progression.py 的 progress() 相同）
// 部队从 Lv.0 起（level_base），城市 / 设施 / 区域从 Lv.1 起；旧服务器没有 level_base 时都按 1 起
export function progress(xp, kind) {
  const th = G.rules.progression[kind];
  let i = 0;
  for (const t of th.slice(1)) if (xp >= t) i++;
  return { xp, level: (G.rules.level_base?.[kind] ?? 1) + i, xp_floor: th[i], xp_next: i + 1 < th.length ? th[i + 1] : null };
}

export function facilityProgress(tile) {
  return progress(G.facilityXp?.get(tile) ?? 0, "facility");
}

// 道路按列还原：a 是差分，b 由"从 a 往哪个方向走一格"推出
function decodeRoads(r) {
  const out = new Array(r.count);
  let a = 0;
  for (let i = 0; i < r.count; i++) {
    a += r.a[i];
    const b = r.dir ? neighbor(a, r.dir[i]) : r.b[i];
    out[i] = { a, b, nation: `nation:${r.nation[i]}`, kind: r.kinds[r.kind[i]], built_turn: r.built[i], water: r.water[i] === 1 };
  }
  return out;
}

// 兼容旧版后端（重启前）发来的 territory 列表
function legacyCityOf(snap, n) {
  const out = new Int32Array(n).fill(-1);
  for (const [cid, tiles] of Object.entries(snap.territory ?? {})) {
    const id = parseInt(cid.slice(5), 10);
    for (const t of tiles) out[t] = id;
  }
  return out;
}

export function setGame(snap, rules, W, H) {
  if (rules) { G.rules = rules; artCache.clear(); artCachePixels=0; scaledArt.clear(); scaledArtPixels=0; }
  G.snap = snap;
  G.W = W;
  G.H = H;
  for (const u of snap.units) Object.assign(u, { col: u.tile % W, row: (u.tile / W) | 0 }, progress(u.xp, "unit"));
  for (const c of snap.cities) Object.assign(c, { col: c.tile % W, row: (c.tile / W) | 0 }, progress(c.xp, "city"));
  if (!Array.isArray(snap.roads)) snap.roads = decodeRoads(snap.roads); // 旧版后端直接发道路列表
  G.nations = new Map(snap.nations.map((n, idx) => [n.id, { ...n, idx, rgb: hexRgb(n.color) }]));
  G.cities = new Map(snap.cities.map((c) => [c.id, c]));
  G.units = new Map(snap.units.map((u) => [u.id, u]));
  G.cityAt = new Map(snap.cities.map((c) => [c.tile, c]));
  G.ownerOf = new Int16Array(W * H).fill(-1);
  G.cityOf = new Int32Array(W * H).fill(-1);
  if (!G.water || G.water.length !== W * H) G.water = new Int8Array(W * H).fill(-1);
  G.owned = [];
  const cityOf = snap.city_of ? unpack(snap.city_of) : legacyCityOf(snap, W * H);
  const nationOfCity = new Map(snap.cities.map((c) => [parseInt(c.id.slice(5), 10), G.nations.get(c.nation).idx]));
  for (let t = 0; t < cityOf.length; t++) {
    const cid = cityOf[t];
    if (cid < 0) continue;
    const idx = nationOfCity.get(cid);
    G.ownerOf[t] = idx;
    G.cityOf[t] = cid;
    G.owned.push([t, idx]);
  }
  G.labels = nationLabels(snap, W);
  G.facilityAt = new Map();
  G.facilityXp = new Map();
  if (snap.facility) {
    const facility = unpack(snap.facility);
    for (let t = 0, k = 0; t < facility.length; t++) {
      if (!facility[t]) continue;
      G.facilityAt.set(t, G.rules.facilities[facility[t] - 1]);
      G.facilityXp.set(t, snap.facility_xp[k++]);
    }
  } else {
    for (const [t, f] of snap.facilities ?? []) {
      G.facilityAt.set(t, G.rules.facilities[f]);
      G.facilityXp.set(t, snap.facility_levels?.[t]?.xp ?? 0);
    }
  }
  G.constructionAt = new Map(snap.cities.filter(c=>c.construction).map(c=>[c.construction.tile,{...c.construction,city:c.id}]));
  G.stacks = new Map();
  G.cargo = new Map(); // 船 / 航母 ID → 船上的陆军（不画在地图上）
  G.territoryOverview = new Map();
  G.roadsAt = new Map();
  for (const road of snap.roads ?? []) {
    for (const t of [road.a, road.b]) {
      if (!G.roadsAt.has(t)) G.roadsAt.set(t, []);
      G.roadsAt.get(t).push(road);
    }
  }
  for (const u of snap.units) {
    if (u.layer === "cargo") {
      if (!G.cargo.has(u.carried_by)) G.cargo.set(u.carried_by, []);
      G.cargo.get(u.carried_by).push(u);
      continue;
    }
    let st = G.stacks.get(u.tile);
    if (!st) G.stacks.set(u.tile, (st = { surface: [], air: [] }));
    st[u.layer === "air" ? "air" : "surface"].push(u);
  }
  for (const st of G.stacks.values()) {
    st.surface.sort(stackOrder);
    st.air.sort(stackOrder);
  }
}

// 国名放在领土的重心（东西环绕：横坐标用圆周平均），字号随领土大小
function nationLabels(snap, W) {
  const PW = SQRT3 * W;
  const acc = snap.nations.map(() => ({ c: 0, s: 0, y: 0, n: 0 }));
  for (const [t, idx] of G.owned) {
    const [x, y] = center(t);
    const a = (2 * Math.PI * x) / PW;
    const k = acc[idx];
    k.c += Math.cos(a);
    k.s += Math.sin(a);
    k.y += y;
    k.n++;
  }
  return snap.nations.map((n, idx) => {
    const k = acc[idx];
    if (!k.n) return null;
    const x = ((Math.atan2(k.s, k.c) / (2 * Math.PI)) * PW + PW) % PW;
    return { name: n.name, rgb: hexRgb(n.color), x, y: k.y / k.n, size: Math.sqrt(k.n) };
  }).filter(Boolean);
}

export function clearGame() {
  G.snap = null;
  G.water = null; // 换了世界，水域缓存作废
  G.selected = null;
  G.actor = G.tactics = G.pending = null;
  G.effects = [];
  G.actionMode = "inspect";
  G.roadPlan = G.roadStart = G.roadNation = null;
}

export function nationOf(id) {
  return G.nations.get(id);
}

export function tileInfo(tile) {
  const st = G.stacks.get(tile) || { surface: [], air: [] };
  return {
    tile,
    col: tile % G.W,
    row: (tile / G.W) | 0,
    city: G.cityAt.get(tile) || null,
    owner: G.ownerOf && G.ownerOf[tile] >= 0 ? G.snap.nations[G.ownerOf[tile]] : null,
    facility: G.facilityAt.get(tile) || null,
    construction: G.constructionAt?.get(tile) || null,
    surface: st.surface,
    air: st.air,
  };
}

// ---------------------------------------------------------------- 选中（只用于查看）

export function select(tile) {
  G.selected = tile == null ? null : { tile };
}

// 单位记着的多回合目的地（随机游走没走到的部分）
function destinationOf(u) {
  return u.destination ? { col: u.destination[0], row: u.destination[1] } : null;
}

// ---------------------------------------------------------------- 绘制工具

function hexPath(ctx, x, y, r) {
  ctx.moveTo(x + CORNERS[0][0] * r, y + CORNERS[0][1] * r);
  for (let i = 1; i < 6; i++) ctx.lineTo(x + CORNERS[i][0] * r, y + CORNERS[i][1] * r);
  ctx.closePath();
}

function roundRect(ctx, x, y, w, h, r) {
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

function onScreen(v, x, y, pad) {
  return x > -pad && x < v.vw + pad && y > -pad && y < v.vh + pad;
}

// 每个可见的东西副本调用 fn(屏幕 x, 屏幕 y)
function eachCopy(v, wx, wy, pad, fn) {
  const y = v.sy(wy);
  if (y < -pad || y > v.vh + pad) return;
  for (let k = v.k0; k <= v.k1; k++) {
    const x = v.sx(wx, k);
    if (x > -pad && x < v.vw + pad) fn(x, y, k);
  }
}

function drawStar(ctx, x, y, r) {
  ctx.beginPath();
  for (let i = 0; i < 10; i++) {
    const a = -Math.PI / 2 + (i * Math.PI) / 5;
    const rr = i % 2 ? r * 0.45 : r;
    ctx.lineTo(x + Math.cos(a) * rr, y + Math.sin(a) * rr);
  }
  ctx.closePath();
}

// ---------------------------------------------------------------- 领土与设施（画在河流之下）

// 某格是不是水域（缓存在 G.water：-1 未知；地形分块没加载时先当陆地，下次再查）
function isWaterTile(v, t) {
  let w = G.water[t];
  if (w < 0) {
    const info = v.tile?.(t % G.W, (t / G.W) | 0);
    if (!info) return false;
    w = G.water[t] = info.water ? 1 : 0;
  }
  return w === 1;
}

function drawTerritory(ctx, v) {
  // The world overview uses a cached atlas; close views visit only visible hexes.
  if (G.snap.stress && v.z < 3 && !v.atlas) {
    let cv = G.territoryOverview.get(G.political);
    if (!cv) {
      cv = document.createElement('canvas');
      cv.width = Math.ceil(v.PW * 2 + 4); cv.height = G.H * 3 + 4;
      const view = { ...v, atlas: true, z: 2, x0: 0, x1: v.PW, y0: 0, y1: G.H * 1.5,
        k0: -1, k1: 1, vw: cv.width, vh: cv.height, sx: (x,k=0)=>(x+k*v.PW)*2+2, sy: y=>y*2+2 };
      const cc = cv.getContext('2d',{willReadFrequently:true});
      const borders = drawTerritory(cc, view); borders();
      G.territoryOverview.set(G.political, cv);
    }
    for (let k=v.k0;k<=v.k1;k++) {
      ctx.drawImage(cv, v.sx(-1,k), v.sy(-1), cv.width*v.z/2, cv.height*v.z/2);
    }
    return null;
  }
  const z = v.z;
  const nations = G.snap.nations.length;
  const fills = Array.from({ length: nations }, () => new Path2D());
  const borders = Array.from({ length: nations }, () => new Path2D());
  const districts = new Path2D(); // 同一国家内不同城市辖区的分界（只在"国家"模式画）
  const political = G.political;
  // 地形模式下领海单独铺得更淡，海面不被国家色染脏
  const seas = political ? null : Array.from({ length: nations }, () => new Path2D());
  const pad = z * 1.2;
  const inset = z < 6 ? 0 : Math.min(0.08, 1.6 / z); // 国界线往格子里收一点，两国的线并排不重叠
  for (const t of visibleTiles(v)) {
    const idx = G.ownerOf[t]; if (idx < 0) continue;
    const [wx, wy] = center(t);
    const fill = seas && isWaterTile(v, t) ? seas[idx] : fills[idx];
    eachCopy(v, wx, wy, pad, (x, y) => {
      hexPath(fill, x, y, z + 0.4);
      for (let d = 0; d < 6; d++) {
        const n = neighbor(t, d);
        if (n >= 0 && G.ownerOf[n] === idx) {
          if (political && z >= 4 && d < 3 && G.cityOf[n] !== G.cityOf[t]) {
            const [a, b] = EDGE_CORNERS[d];
            districts.moveTo(x + CORNERS[a][0] * z, y + CORNERS[a][1] * z);
            districts.lineTo(x + CORNERS[b][0] * z, y + CORNERS[b][1] * z);
          }
          continue;
        }
        const [a, b] = EDGE_CORNERS[d];
        const r = z * (1 - inset);
        borders[idx].moveTo(x + CORNERS[a][0] * r, y + CORNERS[a][1] * r);
        borders[idx].lineTo(x + CORNERS[b][0] * r, y + CORNERS[b][1] * r);
      }
    });
  }
  G.snap.nations.forEach((n, idx) => {
    const c = G.nations.get(n.id).rgb;
    ctx.fillStyle = political ? shade(c, 0.18) : rgba(c, z < 6 ? 0.24 : 0.14);
    ctx.fill(fills[idx]);
    if (seas) {
      ctx.fillStyle = rgba(c, z < 6 ? 0.1 : 0.06);
      ctx.fill(seas[idx]);
    }
  });
  // 国界线返回出去，等城市、设施的地块插画画完再描（否则会被地块盖住）
  return () => {
    ctx.save();
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    const width = political ? Math.max(1.2, Math.min(2.8, z * 0.065)) : Math.max(1.6, Math.min(3, z * 0.07));
    // 地形模式：国界内侧一条国家色色带（裁剪在本国领土里）。领土底色压得很淡以免把地形染脏，
    // 靠这条色带 + 深色描边远看也能分清各国地盘
    const band = z < 6 ? Math.min(5, 2.2 + z * 0.5) : Math.max(5, Math.min(14, z * 0.25));
    G.snap.nations.forEach((n, idx) => {
      const c = G.nations.get(n.id).rgb;
      if (political) {
        ctx.strokeStyle = shade(c, -0.45);
        ctx.lineWidth = width;
        ctx.stroke(borders[idx]);
        return;
      }
      const area = new Path2D(fills[idx]);
      area.addPath(seas[idx]);
      ctx.save();
      ctx.clip(area);
      ctx.strokeStyle = rgba(c, 0.55);
      ctx.lineWidth = band * 2;
      ctx.stroke(borders[idx]);
      ctx.restore();
      ctx.strokeStyle = rgba([20, 16, 10], 0.55);
      ctx.lineWidth = width + 2;
      ctx.stroke(borders[idx]);
      ctx.strokeStyle = shade(c, 0.12);
      ctx.lineWidth = width;
      ctx.stroke(borders[idx]);
    });
    if (political) {
      ctx.setLineDash([Math.max(2, z * 0.2), Math.max(2, z * 0.16)]);
      ctx.strokeStyle = "rgba(40,30,20,0.35)";
      ctx.lineWidth = Math.max(0.8, z * 0.04);
      ctx.stroke(districts);
      ctx.setLineDash([]);
    }
    ctx.restore();
  };
}

// Wrapped row/column bounds keep detailed rendering proportional to the viewport.
function visibleTiles(v) {
  if (v.gameTiles) return v.gameTiles;
  const tiles = [];
  const r0=Math.max(0,Math.floor(v.y0/1.5)-3), r1=Math.min(G.H-1,Math.ceil(v.y1/1.5)+3);
  for(let row=r0;row<=r1;row++) {
    const off=.5*(row&1), c0=Math.floor(v.x0/SQRT3-off)-3;
    const c1=Math.min(c0+G.W-1,Math.ceil(v.x1/SQRT3-off)+3);
    for(let col=c0;col<=c1;col++) tiles.push(row*G.W+((col%G.W)+G.W)%G.W);
  }
  v.gameTiles=tiles;
  return tiles;
}

function visibleEntries(map, v) {
  const tiles=visibleTiles(v);
  return map.size < tiles.length ? map.entries() : tiles.filter(t=>map.has(t)).map(t=>[t,map.get(t)]);
}

// 开发地块铺满六边形，不在城市或设施景观内烘焙道路。
function drawFacilities(ctx, v) {
  const z = v.z;
  if(G.snap.stress && z<28) {
    const fills=new Map();
    for(const [t,fid] of visibleEntries(G.facilityAt,v)) {
      const icon=G.rules.buildings[fid].icon;
      const color=icon==='farm'?'#91a346':TILE_GROUND[icon];
      if(!fills.has(color))fills.set(color,new Path2D());
      const [wx,wy]=center(t);
      eachCopy(v,wx,wy,z*1.8,(x,y)=>hexPath(fills.get(color),x,y,z));
    }
    for(const [color,path] of fills){ctx.fillStyle=color;ctx.fill(path);}
    return;
  }
  for (const [t, fid] of visibleEntries(G.facilityAt,v)) {
    const b = G.rules.buildings[fid];
    const c = G.ownerOf[t] >= 0 ? G.nations.get(G.snap.nations[G.ownerOf[t]].id).rgb : [120, 120, 120];
    const [wx, wy] = center(t);
    const info = v.tile ? v.tile(t % G.W, (t / G.W) | 0) : null;
    eachCopy(v, wx, wy, z * 1.8, (x, y) => {
      drawFacilityTile(ctx, b.icon, x, y, z, c, info, t);
    });
  }
}

function drawCityTiles(ctx, v) {
  const z = v.z;
  for (const [,city] of visibleEntries(G.cityAt,v)) {
    const c = G.nations.get(city.nation).rgb;
    const [wx, wy] = center(city.tile);
    eachCopy(v, wx, wy, z * 1.5, (x, y) => {
      if (z >= (G.snap.stress ? 28 : LOD.tileArt)) {
        drawCityTile(ctx, x, y, z, city, c);
      } else {
        groundFill(ctx, x, y, z, TILE_GROUND.city);
      }
    });
  }
}

export function drawUnder(ctx, v) {
  if (!G.snap) return;
  if (v.mode === "terrain" && !G.political && G.layers.territory) {
    // 观战时把地形底图的饱和度降一些，国家色和兵牌才跳得出来（数据图层的颜色有含义，不动）
    ctx.save();
    ctx.globalCompositeOperation = "saturation";
    ctx.globalAlpha = TERRAIN_MUTE;
    ctx.fillStyle = "#808080";
    ctx.fillRect(0, 0, v.vw, v.vh);
    ctx.restore();
  }
  const smoothing=ctx.imageSmoothingQuality;
  ctx.imageSmoothingQuality='low';
  const strokeBorders = G.layers.territory ? drawTerritory(ctx, v) : null;
  if (v.z >= LOD.tiles) {
    if (G.layers.facilities && !G.political) drawFacilities(ctx, v);
    if (G.layers.cities) drawCityTiles(ctx, v);
  }
  if (strokeBorders) strokeBorders();
  ctx.imageSmoothingQuality=smoothing;
}

// ---------------------------------------------------------------- 地块景观：统一等轴测投影、石材/陶瓦/铜绿配色和左上光源。
// 地面铺满六边形，城市屋顶允许超出地块；国家色只用于旗帜等身份细节。
// z = 格子半径（CSS 像素），info = { role, water, coast, resource }，分块未加载时为 null。

const INK = "rgba(40,36,30,0.3)";
const ISO = 0.5; // 等轴测的纵向压缩
const E1 = [1, 0.5]; // 地面上的两个等轴测方向：往右下 / 往右上
const E2 = [1, -0.5];
const EDGE_ANGLE = [0, -60, -120, 180, 120, 60].map((d) => (d * Math.PI) / 180); // 各方向那条边的外法线
// 远景只铺一层地面色时用的颜色
const TILE_GROUND = {
  city: "#cbbd9b", farm: "#c3b25a", mine: "#9a8b76", lumber: "#667640", plantation: "#6f8a45",
  oil: "#a8997c", factory: "#a29e94", base: "#ae9f74", airfield: "#93a362", port: "#bcb095",
};

function poly(ctx, pts) {
  ctx.beginPath();
  ctx.moveTo(pts[0], pts[1]);
  for (let i = 2; i < pts.length; i += 2) ctx.lineTo(pts[i], pts[i + 1]);
  ctx.closePath();
}
function fillPoly(ctx, pts, fill, lw) {
  poly(ctx, pts);
  ctx.fillStyle = fill;
  ctx.fill();
  if (lw) {
    ctx.strokeStyle = INK;
    ctx.lineWidth = lw;
    ctx.stroke();
  }
}
function blob(ctx, x, y, rx, ry, fill) {
  ctx.beginPath();
  ctx.ellipse(x, y, rx, ry, 0, 0, Math.PI * 2);
  ctx.fillStyle = fill;
  ctx.fill();
}
function line(ctx, pts, color, lw) {
  ctx.beginPath();
  ctx.moveTo(pts[0], pts[1]);
  for (let i = 2; i < pts.length; i += 2) ctx.lineTo(pts[i], pts[i + 1]);
  ctx.strokeStyle = color;
  ctx.lineWidth = lw;
  ctx.stroke();
}
// 描边宽度：格子太小时不描边，免得糊成一团
function lineWidth(z) {
  return z < 18 ? 0 : Math.max(0.35, z * 0.007);
}
function hash32(i) {
  let h = Math.imul(i ^ 0x9e3779b9, 0x85ebca6b);
  h ^= h >>> 13;
  h = Math.imul(h, 0xc2b2ae35);
  return (h ^ (h >>> 16)) >>> 0;
}
// 确定性的小随机数（同一格每次画出来都一样）
function rand(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function inHex(px, py, r) {
  const ax = Math.abs(px);
  return ax <= r * 0.866 && Math.abs(py) <= r - ax * 0.57735;
}
// 六边形内的等轴测网格点（以格子半径为 1 的单位），按从后往前排序
function lattice(step, r, jitter = 0, R = null) {
  const out = [];
  const n = Math.ceil(1.6 / step);
  for (let i = -n; i <= n; i++) {
    for (let j = -n; j <= n; j++) {
      let px = (i + j) * step * E1[0], py = i * step * E1[1] + j * step * E2[1];
      if (jitter && R) {
        px += (R() - 0.5) * jitter * step;
        py += (R() - 0.5) * jitter * step * 0.5;
      }
      if (inHex(px, py, r)) out.push([px, py]);
    }
  }
  return out.sort((a, b) => a[1] - b[1] || a[0] - b[0]);
}
function hexClip(ctx, x, y, r) {
  ctx.beginPath();
  for (let i = 0; i < 6; i++) {
    const a = ((60 * i - 30) * Math.PI) / 180;
    ctx.lineTo(x + Math.cos(a) * r, y + Math.sin(a) * r);
  }
  ctx.closePath();
  ctx.clip();
}
function groundFill(ctx, x, y, z, color) {
  ctx.beginPath(); hexPath(ctx, x, y, z);
  ctx.fillStyle = color; ctx.fill();
}
function speckle(ctx, x, y, z, R, n, colors, size) {
  for (let k = 0; k < n; k++) {
    const px = (R() - 0.5) * 1.8, py = (R() - 0.5) * 2;
    if (!inHex(px, py, 1)) continue;
    blob(ctx, x + px * z, y + py * z, size * z * (0.5 + R()), size * z * 0.5 * (0.5 + R()), colors[k % colors.length]);
  }
}

// ---- 建筑部件

// 等轴测方盒：(x, y) 是前角落地点，a = 前角到左角（沿 −E1），b = 前角到右角（沿 E2），h = 墙高
function isoBox(ctx, x, y, a, h, wallL, wallR, lw, b = a) {
  const ka = a * ISO, kb = b * ISO;
  // 阴影朝右下投射，与屋顶的左上光源一致。
  const dx = h * 0.55, dy = h * 0.3;
  fillPoly(ctx, [x - a, y - ka, x - a + b, y - ka - kb, x + b + dx, y - kb + dy, x + dx, y + dy], "rgba(28,32,25,0.22)", 0);
  const left = ctx.createLinearGradient(x, y - h, x, y);
  left.addColorStop(0, wallL); left.addColorStop(1, shade(hexRgb(wallL), -0.12));
  const right = ctx.createLinearGradient(x, y - h, x + b, y);
  right.addColorStop(0, wallR); right.addColorStop(1, shade(hexRgb(wallR), -0.16));
  fillPoly(ctx, [x - a, y - ka, x, y, x, y - h, x - a, y - ka - h], left, lw);
  fillPoly(ctx, [x, y, x + b, y - kb, x + b, y - kb - h, x, y - h], right, lw);
}
function flatRoof(ctx, x, y, a, h, fill, lw, b = a) {
  const ka = a * ISO, kb = b * ISO;
  fillPoly(ctx, [x - a, y - ka - h, x, y - h, x + b, y - kb - h, x - a + b, y - ka - kb - h], fill, lw);
}
// 四坡屋顶（方形平面）
function hipRoof(ctx, x, y, a, h, r, roofL, roofR, lw) {
  const k = a * ISO, o = a * 0.16;
  const ax = x, ay = y - k - h - r;
  fillPoly(ctx, [x - a - o, y - k - h + o * 0.3, x, y - h + o * 0.6, ax, ay], roofL, lw);
  fillPoly(ctx, [x, y - h + o * 0.6, x + a + o, y - k - h + o * 0.3, ax, ay], roofR, lw);
  line(ctx, [x - a - o, y - k - h, ax, ay], "rgba(255,242,209,0.4)", Math.max(0.3, a * 0.035));
}
// 两坡屋顶：屋脊沿左墙方向（长边 a），前右端是山墙
function gableRoof(ctx, x, y, a, b, h, r, roofL, roofR, gable, lw) {
  const ka = a * ISO, kb = b * ISO;
  const F = [x, y - h], L = [x - a, y - ka - h], Rr = [x + b, y - kb - h], B = [x - a + b, y - ka - kb - h];
  const P1 = [(F[0] + Rr[0]) / 2, (F[1] + Rr[1]) / 2 - r], P2 = [(L[0] + B[0]) / 2, (L[1] + B[1]) / 2 - r];
  fillPoly(ctx, [P2[0], P2[1], P1[0], P1[1], Rr[0], Rr[1], B[0], B[1]], roofR, lw);
  fillPoly(ctx, [L[0], L[1], F[0], F[1], P1[0], P1[1], P2[0], P2[1]], roofL, lw);
  fillPoly(ctx, [F[0], F[1], Rr[0], Rr[1], P1[0], P1[1]], gable, lw);
}
function windows(ctx, x, y, a, h, color) {
  if (a < 4) return;
  const k = a * ISO;
  ctx.fillStyle = color;
  for (const [t1, t2] of [[0.22, 0.4], [0.6, 0.78]]) {
    const p = (t) => [x - a + a * t, y - k + k * t];
    const [x1, y1] = p(t1), [x2, y2] = p(t2);
    poly(ctx, [x1, y1 - h * 0.35, x2, y2 - h * 0.35, x2, y2 - h * 0.68, x1, y1 - h * 0.68]);
    ctx.fill();
  }
}
function flag(ctx, x, y, len, c, lw) {
  line(ctx, [x, y, x, y - len], "#3a2e24", Math.max(0.8, len * 0.05));
  const fw = len * 0.55, fh = len * 0.34, top = y - len;
  ctx.beginPath();
  ctx.moveTo(x, top);
  ctx.quadraticCurveTo(x + fw * 0.5, top - fh * 0.25, x + fw, top + fh * 0.1);
  ctx.lineTo(x + fw, top + fh * 1.05);
  ctx.quadraticCurveTo(x + fw * 0.5, top + fh * 0.75, x, top + fh);
  ctx.closePath();
  ctx.fillStyle = shade(c, 0.05);
  ctx.fill();
  if (lw) {
    ctx.strokeStyle = INK;
    ctx.lineWidth = lw;
    ctx.stroke();
  }
}
function roundTree(ctx, x, y, r, dark) {
  blob(ctx, x + r * 0.3, y + r * 0.2, r * 0.95, r * 0.4, "rgba(20,30,15,0.25)");
  line(ctx, [x, y, x, y - r * 0.6], "#5a4030", Math.max(0.6, r * 0.18));
  ctx.beginPath();
  ctx.arc(x, y - r * 0.9, r, 0, Math.PI * 2);
  ctx.fillStyle = dark ? "#3a5a2f" : "#4f7536";
  ctx.fill();
  blob(ctx, x - r * 0.3, y - r * 1.2, r * 0.42, r * 0.42, "rgba(185,215,140,0.4)");
}
function conifer(ctx, x, y, s) {
  blob(ctx, x + s * 0.06, y + s * 0.02, s * 0.14, s * 0.05, "rgba(20,30,15,0.25)");
  line(ctx, [x, y, x, y - s * 0.1], "#5a4030", Math.max(0.6, s * 0.04));
  fillPoly(ctx, [x - s * 0.13, y - s * 0.07, x, y - s * 0.55, x, y - s * 0.07], "#4f7d3a", 0);
  fillPoly(ctx, [x, y - s * 0.07, x, y - s * 0.55, x + s * 0.13, y - s * 0.07], "#33572a", 0);
}
function house(ctx, x, y, a, h, roof, lw) {
  isoBox(ctx, x, y, a, h, "#e9dcc0", "#b9aa91", lw);
  windows(ctx, x, y, a, h, "#49525a");
  if (a >= 5) {
    fillPoly(ctx, [x + a * 0.18, y - a * 0.09, x + a * 0.18, y - h * 0.48 - a * 0.09, x + a * 0.36, y - h * 0.48 - a * 0.18, x + a * 0.36, y - a * 0.18], "#655744", 0);
  }
  hipRoof(ctx, x, y, a, h, a * 0.48, roof[0], roof[1], lw);
}
function tent(ctx, x, y, w, h, lw) {
  const d = [w * 0.9, -w * 0.45];
  fillPoly(ctx, [x + w, y, x, y - h, x + d[0], y - h + d[1], x + w + d[0], y + d[1]], "#5f6a35", lw);
  fillPoly(ctx, [x - w, y, x, y - h, x + w, y], "#97a45f", lw);
  fillPoly(ctx, [x - w * 0.3, y, x, y - h * 0.45, x + w * 0.3, y], "#3c4322", 0);
}
// 带锯齿天窗屋顶的厂房，后面立两根冒烟的烟囱
function factoryHall(ctx, bx, by, a, h, z, lw) {
  const k = a * ISO;
  for (const [t, ch] of [[0.2, 0.55], [0.65, 0.45]]) {
    const cx = bx - a + a * t + a * 1.05, cy = by - k + k * t - k * 1.05;
    const H = ch * z + h;
    isoBox(ctx, cx, cy, z * 0.045, H, "#9b4b35", "#6b3123", lw);
    ctx.fillStyle = "#ece6da";
    ctx.fillRect(cx - z * 0.045, cy - H * 0.86, z * 0.09, z * 0.035);
    flatRoof(ctx, cx, cy, z * 0.045, H, "#3d2a22", lw);
    for (const [px, py, r] of [[0.05, 0.1, 0.06], [0.14, 0.2, 0.08], [0.27, 0.3, 0.1]]) {
      blob(ctx, cx + px * z, cy - H - py * z, r * z, r * z * 0.85, "rgba(236,234,228,0.85)");
    }
  }
  isoBox(ctx, bx, by, a, h, "#cf8460", "#985539", lw);
  windows(ctx, bx, by, a, h, "rgba(60,40,30,0.55)");
  flatRoof(ctx, bx, by, a, h, "#8a8e90", lw);
  const Lx = bx - a, Ly = by - k - h, tz = z * 0.08;
  for (let i = 0; i < 3; i++) {
    const t0 = i / 3, t1 = (i + 1) / 3;
    const x0 = Lx + a * t0, y0 = Ly + k * t0, x1 = Lx + a * t1, y1 = Ly + k * t1;
    fillPoly(ctx, [x0, y0, x1, y1 - tz, x1 + a, y1 - k - tz, x0 + a, y0 - k], "#7a7e80", lw);
    fillPoly(ctx, [x1, y1, x1, y1 - tz, x1 + a, y1 - k - tz, x1 + a, y1 - k], "#bcd8e2", lw);
  }
}

// ---- 城市

const ROOF_TERRACOTTA = ["#ad7355", "#764935"];
const ROOF_SLATE = ["#758a8b", "#465f64"];

function churchTower(ctx, x, y, z, c, lw) {
  const a = z * 0.09, h = z * 0.42;
  isoBox(ctx, x, y, a, h, "#f4ead3", "#cbb791", lw);
  hipRoof(ctx, x, y, a, h, z * 0.24, shade(c, 0.12), shade(c, -0.3), lw);
}
function wallEdge(ctx, x, y, z, i, lw) {
  const r = z * 0.9;
  const a0 = ((60 * i - 30) * Math.PI) / 180, a1 = ((60 * (i + 1) - 30) * Math.PI) / 180;
  const p0 = [x + Math.cos(a0) * r, y + Math.sin(a0) * r], p1 = [x + Math.cos(a1) * r, y + Math.sin(a1) * r];
  const hh = z * 0.09, t = z * 0.05;
  fillPoly(ctx, [p0[0], p0[1], p1[0], p1[1], p1[0], p1[1] - hh, p0[0], p0[1] - hh], "#a59d90", lw);
  fillPoly(ctx, [p0[0], p0[1] - hh, p1[0], p1[1] - hh, p1[0], p1[1] - hh - t * 0.6, p0[0], p0[1] - hh - t * 0.6], "#cfc8bb", lw);
}
function wallTower(ctx, x, y, z, i, lw) {
  const a = ((60 * i - 30) * Math.PI) / 180;
  const tx = x + Math.cos(a) * z * 0.9, ty = y + Math.sin(a) * z * 0.9 + z * 0.04;
  isoBox(ctx, tx, ty, z * 0.06, z * 0.16, "#bdb5a8", "#8a8278", lw);
  flatRoof(ctx, tx, ty, z * 0.06, z * 0.16, "#d6cfc3", lw);
}

function civicHall(ctx, x, y, z, c, lw, capital) {
  const a = z * 0.25, h = z * 0.22;
  // 基座、两侧翼楼、中央门廊与铜绿色圆顶。
  fillPoly(ctx, [x - a * 1.35, y - a * 0.45, x, y + a * 0.25, x + a * 1.35, y - a * 0.45, x, y - a * 1.12], "#d7cdb4", lw);
  for (const [px, py] of [[-0.19, -0.095], [0.19, -0.095]]) {
    isoBox(ctx, x + px * z, y + py * z, z * 0.13, h * 0.78, "#eee3c9", "#b5a98c", lw);
    flatRoof(ctx, x + px * z, y + py * z, z * 0.13, h * 0.78, "#657a78", lw);
    windows(ctx, x + px * z, y + py * z, z * 0.13, h * 0.78, "#4a5b61");
  }
  isoBox(ctx, x, y, a * 0.66, h, "#f5e9d0", "#c4b598", lw);
  flatRoof(ctx, x, y, a * 0.66, h, "#ede1c8", lw);
  const cy = y - a * 0.33 - h, rx = z * 0.13, ry = z * 0.17;
  const dome = ctx.createLinearGradient(x - rx, cy - ry, x + rx, cy);
  dome.addColorStop(0, "#a6c2ac"); dome.addColorStop(0.45, "#668d80"); dome.addColorStop(1, "#355a54");
  ctx.beginPath(); ctx.ellipse(x, cy, rx, ry, 0, Math.PI, Math.PI * 2);
  ctx.closePath(); ctx.fillStyle = dome; ctx.fill();
  blob(ctx, x, cy, rx, z * 0.025, "#506f64");
  line(ctx, [x, cy - ry, x, cy - ry - z * 0.06], "#d4bd85", z * 0.012);
  // 柱廊留出真正的暗部，不用粗描边勾建筑。
  for (let i = -1; i <= 1; i++) {
    const px = x - a * 0.52 + (i + 1) * a * 0.2;
    const py = y - a * 0.26 + (i + 1) * a * 0.1;
    fillPoly(ctx, [px, py - h * 0.13, px, py - h * 0.76, px + z * 0.025, py - h * 0.76 + z * 0.012, px + z * 0.025, py - h * 0.13 + z * 0.012], "#766b59", 0);
  }
  for (let i = 0; i < 3; i++) line(ctx, [x - a * 0.65, y + z * (0.025 + i * 0.018), x, y + a * 0.32 + z * i * 0.018], "#e6dac0", z * 0.018);
  if (capital) flag(ctx, x + z * 0.28, y - z * 0.12, z * 0.47, c, lw);
}

function drawCityScene(ctx, x, y, z, city, c) {
  const lw = lineWidth(z);
  const R = rand(hash32(parseInt(city.id.slice(5), 10)) ^ 0x51ed27);
  const tier = city.population <= 2 ? 0 : city.population <= 5 ? 1 : 2;
  const walls = city.buildings.includes("walls");
  ctx.save();
  ctx.lineCap = "round"; ctx.lineJoin = "round";
  // 城区地面占满格子，只限制地面，允许旗杆、屋顶伸出格子。
  ctx.save(); hexClip(ctx, x, y, z);
  groundFill(ctx, x, y, z, "#b6b092");
  blob(ctx, x, y + z * 0.08, z * 0.69, z * 0.46, "rgba(210,199,169,0.65)");
  fillPoly(ctx, [x - z * 0.36, y + z * 0.16, x, y + z * 0.34, x + z * 0.36, y + z * 0.16, x, y - z * 0.02], "#e0d4b5", 0);
  ctx.restore();
  if (walls) {
    for (const i of [3, 4, 5, 0]) wallEdge(ctx, x, y, z * 0.88, i, lw);
    for (const i of [4, 5, 0]) wallTower(ctx, x, y, z * 0.88, i, lw);
  }
  // 稀疏、可读的街区；人口增长增加街区和层数，不改变已有房屋的位置。
  const slots = [
    [-0.41, -0.25], [0.03, -0.43], [0.48, -0.2], [-0.54, 0.09],
    [0.46, 0.2], [-0.3, 0.42], [0.05, 0.53], [0.32, 0.48],
    [-0.25, -0.43], [0.29, -0.37], [-0.58, 0.32], [0.61, 0.04],
    [-0.04, 0.35], [-0.42, -0.02], [0.61, 0.36], [-0.15, -0.13],
  ];
  const items = slots.map(([px, py], i) => ({
    px, py, kind: "house", visible: i < [8, 12, 16][tier],
    a: z * (0.115 + R() * 0.025), h: z * (0.11 + R() * 0.035 + tier * 0.035),
    roof: R() < 0.65 ? ROOF_TERRACOTTA : ROOF_SLATE,
  })).filter(it => it.visible);
  // 市中心建筑也反映实际建造结果。
  if (city.buildings.includes("workshop")) items.push({ px: -0.48, py: 0.12, kind: "workshop" });
  if (city.buildings.includes("market")) items.push({ px: 0.08, py: 0.39, kind: "market" });
  if (city.buildings.includes("granary")) items.push({ px: 0.46, py: 0.2, kind: "granary" });
  if (city.buildings.includes("school")) items.push({ px: 0.02, py: -0.43, kind: "school" });
  if (city.buildings.includes("barracks")) items.push({ px: -0.3, py: 0.42, kind: "barracks" });
  // 功能建筑替换所在街区，避免多个建筑画在同一位置。
  const special = items.filter(it => it.kind !== "house");
  const scene = items.filter(it => it.kind !== "house" || !special.some(s => Math.hypot(s.px - it.px, s.py - it.py) < 0.15));
  scene.push({ px: 0, py: 0.11, kind: "hall" });
  scene.push({ px: -0.65, py: -0.14, kind: "tree" }, { px: 0.66, py: 0.22, kind: "tree" });
  scene.sort((a, b) => a.py - b.py || a.px - b.px);
  for (const it of scene) {
    const bx = x + it.px * z, by = y + it.py * z;
    if (it.kind === "hall") civicHall(ctx, bx, by, z, c, lw, city.capital);
    else if (it.kind === "tree") roundTree(ctx, bx, by, z * 0.055, false);
    else if (it.kind === "workshop") factoryHall(ctx, bx, by, z * 0.14, z * 0.12, z * 0.48, lw);
    else if (it.kind === "market") {
      for (const [off, col] of [[-0.12, "#ad6049"], [0.12, "#c4ad73"]]) {
        isoBox(ctx, bx + off * z, by, z * 0.08, z * 0.07, "#e0d0aa", "#b3a07c", lw);
        gableRoof(ctx, bx + off * z, by, z * 0.09, z * 0.09, z * 0.07, z * 0.045, col, "#786650", "#bca67f", lw);
      }
    } else if (it.kind === "granary" || it.kind === "barracks") {
      isoBox(ctx, bx, by, z * 0.16, z * 0.12, "#d4c8aa", "#9e9177", lw, z * 0.11);
      gableRoof(ctx, bx, by, z * 0.16, z * 0.11, z * 0.12, z * 0.07, "#8b8270", "#5d655e", "#c4b493", lw);
    } else if (it.kind === "school") {
      house(ctx, bx, by, z * 0.17, z * 0.2, ROOF_SLATE, lw);
      churchTower(ctx, bx, by - z * 0.05, z * 0.35, [100, 123, 122], lw);
    } else house(ctx, bx, by, it.a, it.h, it.roof, lw);
  }
  if (walls) {
    for (const i of [1, 2]) wallEdge(ctx, x, y, z * 0.88, i, lw);
    for (const i of [1, 2, 3]) wallTower(ctx, x, y, z * 0.88, i, lw);
  }
  ctx.restore();
}

// 用固定分辨率烘焙景观：缩放时材质和细节稳定，避免每帧重复画数百个多边形。
const ART_RADIUS = 96;
const ART_CACHE_PIXELS = 32_000_000;
const artCache = new Map();
let artCachePixels = 0;
const scaledArt = new Map();
let scaledArtPixels = 0;
function tileArtwork(ctx, key, x, y, z, paint, radius = ART_RADIUS) {
  // Medium zoom needs more sprites, at a smaller resolution. A pixel budget
  // prevents the visible set from evicting itself and repainting every frame.
  if(z<48)radius=Math.min(radius,64);
  key+=`:${radius}`;
  let canvas = artCache.get(key);
  if (!canvas) {
    canvas = document.createElement("canvas"); canvas.width = canvas.height = radius * 3;
    paint(canvas.getContext("2d",{willReadFrequently:true}), canvas.width / 2, canvas.height / 2, radius);
    artCache.set(key, canvas);
    artCachePixels+=canvas.width*canvas.height;
    while(artCachePixels>ART_CACHE_PIXELS && artCache.size>1) {
      const oldest=artCache.keys().next().value,old=artCache.get(oldest);
      artCachePixels-=old.width*old.height;artCache.delete(oldest);
    }
  }else {
    artCache.delete(key);artCache.set(key,canvas);
  }
  const transform=ctx.getTransform(),dpr=Math.hypot(transform.a,transform.b);
  const size=Math.ceil(z*3*dpr);
  if(size<canvas.width) {
    const scaledKey=`${key}:${size}`;
    let scaled=scaledArt.get(scaledKey);
    if(!scaled) {
      scaled=document.createElement('canvas');scaled.width=scaled.height=size;
      const cc=scaled.getContext('2d',{willReadFrequently:true});
      cc.imageSmoothingQuality='high';cc.drawImage(canvas,0,0,size,size);
      scaledArt.set(scaledKey,scaled);scaledArtPixels+=size*size;
      while(scaledArtPixels>8_000_000&&scaledArt.size>1) {
        const oldest=scaledArt.keys().next().value,old=scaledArt.get(oldest);
        scaledArtPixels-=old.width*old.height;scaledArt.delete(oldest);
      }
    }else {scaledArt.delete(scaledKey);scaledArt.set(scaledKey,scaled);}
    canvas=scaled;
  }
  ctx.drawImage(canvas, x - z * 1.5, y - z * 1.5, z * 3, z * 3);
}

export function drawCityTile(ctx, x, y, z, city, c) {
  const key = `city:${city.id}:${city.population <= 2 ? 0 : city.population <= 5 ? 1 : 2}:${city.capital}:${city.buildings.join(",")}:${c}`;
  tileArtwork(ctx, key, x, y, z, (cx, px, py, s) => drawCityScene(cx, px, py, s, city, c));
}

// ---- 城外设施

function cropField(ctx, z, R, angle, palette) {
  ctx.save(); ctx.rotate(angle);
  const spacing = z * 0.12, phase = R() * Math.PI * 2;
  const bend = (t, row) => z * (Math.sin(t * 2 + phase + row * 0.12) * 0.012 + Math.sin(t * 9 + row) * 0.003);
  for (let row = -16; row <= 16; row++) {
    const ox = row * spacing, width = spacing * (0.86 + R() * 0.05);
    const ridge = ctx.createLinearGradient(ox - width * 0.5, 0, ox + width * 0.5, 0);
    ridge.addColorStop(0, palette[0]); ridge.addColorStop(0.25, palette[2]);
    ridge.addColorStop(0.65, palette[1]); ridge.addColorStop(1, palette[0]);
    const left = [], right = [];
    for (let i = 0; i <= 32; i++) {
      const t = -1.9 + i * 3.8 / 32, bx = ox + bend(t, row);
      left.push(bx - width * 0.5, t * z); right.unshift(bx + width * 0.5, t * z);
    }
    ctx.save(); poly(ctx, [...left, ...right]); ctx.clip();
    ctx.fillStyle = ridge; ctx.fillRect(ox - spacing, -z * 2, spacing * 2, z * 4);
    for (let i = 0; i < 90; i++) {
      const t = (R() - 0.5) * 3.8;
      const bx = ox + bend(t, row) + (R() - 0.5) * width * 0.86, by = t * z;
      const len = z * (0.011 + R() * 0.016);
      ctx.strokeStyle = i % 2 ? "rgba(207,218,137,0.35)" : "rgba(58,83,32,0.22)";
      ctx.lineWidth = z * (0.004 + R() * 0.003);
      ctx.beginPath(); ctx.moveTo(bx, by); ctx.lineTo(bx - len * 0.3, by - len); ctx.stroke();
    }
    ctx.restore();
  }
  ctx.restore();
}

function farmTile(ctx, x, y, z, R) {
  groundFill(ctx, x, y, z, "#96a44e");
  // 三至四块相接的田地各有垄向，分界没有道路或空地。
  const splitX = (R() - 0.5) * 0.34 * z, splitY = (R() - 0.5) * 0.4 * z;
  const four = R() < 0.55, rotation = Math.floor(R() * 4);
  const layoutAngle = (R() - 0.5) * 0.55;
  const palettes = [["#738839", "#99ab4f", "#b4c16b"], ["#637d35", "#86a148", "#a2b75d"], ["#838d43", "#b1b45c", "#c4c575"], ["#6d843b", "#92a650", "#aebb69"]];
  const plots = four ? [
    [-1.6 * z, -1.6 * z, splitX, splitY], [splitX, -1.6 * z, 1.6 * z, splitY],
    [-1.6 * z, splitY, splitX, 1.6 * z], [splitX, splitY, 1.6 * z, 1.6 * z],
  ] : [
    [-1.6 * z, -1.6 * z, 1.6 * z, splitY],
    [-1.6 * z, splitY, splitX, 1.6 * z], [splitX, splitY, 1.6 * z, 1.6 * z],
  ];
  ctx.save(); ctx.translate(x, y); ctx.rotate(layoutAngle);
  plots.forEach(([x0, y0, x1, y1], i) => {
    const style = (i + rotation) % 4;
    ctx.save(); ctx.beginPath(); ctx.rect(x0, y0, x1 - x0, y1 - y0); ctx.clip();
    ctx.fillStyle = palettes[style][1]; ctx.fillRect(x0, y0, x1 - x0, y1 - y0);
    cropField(ctx, z, R, [0, Math.PI / 2, Math.PI / 3, -Math.PI / 3][style] + (R() - 0.5) * 0.1, palettes[style]);
    ctx.restore();
  });
  ctx.restore();
  speckle(ctx, x, y, z, R, 120, ["rgba(215,214,145,0.2)", "rgba(65,91,35,0.14)"], 0.005);
}

function quarryRock(ctx, x, y, s, color) {
  const c = hexRgb(color);
  blob(ctx, x + s * 0.12, y + s * 0.08, s * 0.72, s * 0.26, "rgba(42,48,32,0.15)");
  fillPoly(ctx, [x - s * 0.65, y, x - s * 0.3, y - s * 0.45, x + s * 0.05, y - s * 0.55, x + s * 0.17, y - s * 0.1, x, y + s * 0.17], shade(c, 0.08), 0);
  fillPoly(ctx, [x, y + s * 0.17, x + s * 0.17, y - s * 0.1, x + s * 0.05, y - s * 0.55, x + s * 0.45, y - s * 0.31, x + s * 0.67, y + s * 0.04], shade(c, -0.16), 0);
  line(ctx, [x - s * 0.3, y - s * 0.45, x + s * 0.05, y - s * 0.55, x + s * 0.45, y - s * 0.31], "rgba(221,220,188,0.38)", s * 0.02);
}

function mineTile(ctx, x, y, z, R, c, info) {
  const lw = lineWidth(z), terrain = info?.color ?? "#a9a27c";
  groundFill(ctx, x, y, z, terrain);
  // 保留整格地形色与植被；开采面只占岩坡的一部分，不再画巨大的同心矿坑。
  const ground = ctx.createRadialGradient(x + z * 0.1, y + z * 0.1, z * 0.18, x + z * 0.1, y + z * 0.1, z * 0.76);
  ground.addColorStop(0, "rgba(165,159,128,0.76)"); ground.addColorStop(1, "rgba(165,159,128,0)");
  ctx.fillStyle = ground; ctx.fillRect(x - z, y - z, z * 2, z * 2);
  speckle(ctx, x, y, z, R, 80, ["rgba(229,218,183,0.22)", "rgba(75,83,49,0.17)"], 0.016);
  for (const [px, py, size] of [[-0.4,-0.17,0.5],[-0.05,-0.29,0.46],[0.26,-0.14,0.34]]) {
    quarryRock(ctx, x + px * z, y + py * z, size * z, "#999a82");
  }
  // 裸露的两层岩壁融入山坡，前侧有木支架的矿洞。
  const qx = x - z * 0.16, qy = y + z * 0.18;
  fillPoly(ctx, [qx - z * 0.39,qy - z * 0.06,qx - z * 0.21,qy - z * 0.24,qx + z * 0.23,qy - z * 0.19,qx + z * 0.34,qy + z * 0.02,qx + z * 0.08,qy + z * 0.15], "#a7a690", 0);
  fillPoly(ctx, [qx - z * 0.3,qy + z * 0.02,qx - z * 0.17,qy - z * 0.13,qx + z * 0.2,qy - z * 0.1,qx + z * 0.25,qy + z * 0.1,qx + z * 0.06,qy + z * 0.18], "#858976", 0);
  line(ctx, [qx - z * 0.28,qy - z * 0.04,qx - z * 0.16,qy - z * 0.15,qx + z * 0.2,qy - z * 0.12], "rgba(216,209,173,0.52)", z * 0.018);
  const mx = qx - z * 0.025, my = qy + z * 0.13;
  fillPoly(ctx, [mx - z * 0.12,my,mx - z * 0.1,my - z * 0.15,mx + z * 0.07,my - z * 0.2,mx + z * 0.12,my - z * 0.045,mx + z * 0.1,my + z * 0.04], "#414b3d", 0);
  line(ctx, [mx - z * 0.12,my,mx - z * 0.1,my - z * 0.16,mx + z * 0.07,my - z * 0.21,mx + z * 0.12,my - z * 0.04], "#a78c61", z * 0.025);
  // 规模较小的井架、矿棚与矿石堆，不遮住整片山坡。
  const hx = x + z * 0.4, hy = y + z * 0.16;
  isoBox(ctx, hx + z * 0.07, hy + z * 0.08, z * 0.14, z * 0.1, "#b8a989", "#867d64", lw, z * 0.09);
  gableRoof(ctx, hx + z * 0.07, hy + z * 0.08, z * 0.14, z * 0.09, z * 0.1, z * 0.06, "#79847a", "#52655d", "#a59c7b", lw);
  const tx = hx - z * 0.12, ty = hy - z * 0.07;
  line(ctx, [tx - z * 0.08,ty,tx - z * 0.025,ty - z * 0.34,tx + z * 0.065,ty - z * 0.02], "#76664a", z * 0.02);
  line(ctx, [tx - z * 0.06,ty - z * 0.12,tx + z * 0.035,ty - z * 0.12], "#76664a", z * 0.016);
  blob(ctx, tx - z * 0.025, ty - z * 0.34, z * 0.035, z * 0.035, "#8f876f");
  const ore = info?.resource === "coal" ? "#626a58" : info?.resource === "iron" ? "#958a75" : "#ada67c";
  for (const [px, py, size] of [[-0.49,0.37,0.1],[-0.31,0.48,0.13],[0.38,0.5,0.11],[0.56,0.4,0.07]]) {
    quarryRock(ctx, x + px * z, y + py * z, size * z, ore);
  }
  for (const [px, py] of [[-0.57,-0.38],[0.42,-0.42],[-0.62,0.2],[0.14,0.62],[-0.12,-0.66]]) {
    const bx = x + px * z, by = y + py * z;
    blob(ctx, bx, by, z * 0.08, z * 0.04, "rgba(123,139,68,0.26)");
    for (let j = -1; j <= 1; j++) line(ctx, [bx + j*z*0.025,by,bx + j*z*0.035,by - z*0.048], "rgba(98,117,57,0.58)", z * 0.012);
  }
}

function lumberTile(ctx, x, y, z, R) {
  const lw = lineWidth(z);
  groundFill(ctx, x, y, z, "#6a7a42");
  speckle(ctx, x, y, z, R, 30, ["rgba(40,55,25,0.35)", "rgba(160,170,100,0.3)"], 0.03);
  // 前面伐开的空地和树桩
  blob(ctx, x + z * 0.05, y + z * 0.3, z * 0.62, z * 0.32, "#8e8a55");
  const items = [];
  for (const [px, py] of lattice(0.32, 0.9, 0.5, R)) {
    const cleared = py > -0.08 && Math.abs(px) < 0.62;
    if (cleared) {
      if (R() < 0.35) items.push({ px, py, kind: "stump" });
    } else items.push({ px, py, kind: R() < 0.6 ? "conifer" : "tree" });
  }
  items.push({ px: 0.2, py: 0.24, kind: "mill" });
  items.push({ px: -0.28, py: 0.42, kind: "logs" });
  items.sort((p, q) => p.py - q.py || p.px - q.px);
  for (const it of items) {
    const bx = x + it.px * z, by = y + it.py * z;
    if (it.kind === "conifer") conifer(ctx, bx, by, z * 0.55);
    else if (it.kind === "tree") roundTree(ctx, bx, by, z * 0.09, R() < 0.5);
    else if (it.kind === "stump") {
      blob(ctx, bx, by, z * 0.035, z * 0.018, "#7a5534");
      blob(ctx, bx, by - z * 0.01, z * 0.03, z * 0.015, "#d8b47c");
    } else if (it.kind === "mill") {
      isoBox(ctx, bx, by, z * 0.22, z * 0.13, "#c99d68", "#93683f", lw, z * 0.15);
      gableRoof(ctx, bx, by, z * 0.22, z * 0.15, z * 0.13, z * 0.09, "#7a5a40", "#523b29", "#a97a4a", lw);
    } else {
      const r = z * 0.045;
      blob(ctx, bx + z * 0.03, by + z * 0.01, z * 0.17, z * 0.05, "rgba(30,24,18,0.22)");
      for (const [row, n] of [[0, 3], [1, 2], [2, 1]]) {
        for (let i = 0; i < n; i++) {
          const lx = bx + (i - (n - 1) / 2) * r * 2.05, ly = by - r - row * r * 1.75;
          ctx.beginPath();
          ctx.arc(lx, ly, r, 0, Math.PI * 2);
          ctx.fillStyle = "#e6c38c";
          ctx.fill();
          ctx.strokeStyle = "#8a5a35";
          ctx.lineWidth = Math.max(0.6, z * 0.015);
          ctx.stroke();
        }
      }
    }
  }
}

const PLANTATION_ACCENT = { spices: "#d0483a", wine: "#6b3a7a", silk: "#f3eee2", rubber: null };

function plantationTile(ctx, x, y, z, R, c, info) {
  const lw = lineWidth(z);
  groundFill(ctx, x, y, z, "#7d8b47");
  const accent = info && info.resource in PLANTATION_ACCENT ? PLANTATION_ACCENT[info.resource] : "#e0a43a";
  // 垄沟（沿等轴测方向的一条条土垄）
  ctx.strokeStyle = "#8c6d45";
  ctx.lineWidth = z * 0.07;
  ctx.beginPath();
  for (let k = -6; k <= 6; k++) {
    const oy = y + k * 0.2 * z;
    ctx.moveTo(x - z * 1.2, oy - z * 0.6);
    ctx.lineTo(x + z * 1.2, oy + z * 0.6);
  }
  ctx.stroke();
  const rubber = info?.resource === "rubber";
  const spots = lattice(0.2, 0.95).filter(([px, py]) => !(px > 0.25 && py < -0.15 && py > -0.6));
  for (const [px, py] of spots) {
    const bx = x + px * z, by = y + py * z;
    if (rubber) {
      roundTree(ctx, bx, by, z * 0.07, R() < 0.4);
      continue;
    }
    const r = z * 0.06;
    blob(ctx, bx + r * 0.3, by + r * 0.5, r, r * 0.4, "rgba(20,30,15,0.25)");
    ctx.beginPath();
    ctx.arc(bx, by - r * 0.3, r, 0, Math.PI * 2);
    ctx.fillStyle = "#3f7536";
    ctx.fill();
    blob(ctx, bx - r * 0.3, by - r * 0.6, r * 0.4, r * 0.4, "rgba(170,210,120,0.5)");
    if (accent) {
      blob(ctx, bx + r * 0.35, by - r * 0.2, r * 0.22, r * 0.22, accent);
      blob(ctx, bx - r * 0.25, by + r * 0.05, r * 0.18, r * 0.18, accent);
    }
  }
  // 后面的晾晒棚
  const sx = x + z * 0.45, sy = y - z * 0.32;
  isoBox(ctx, sx, sy, z * 0.14, z * 0.09, "#e2d2ad", "#b39d72", lw, z * 0.1);
  gableRoof(ctx, sx, sy, z * 0.14, z * 0.1, z * 0.09, z * 0.06, "#8a6a45", "#634a2f", "#c9b48a", lw);
}

function pumpjack(ctx, x, y, s, lw) {
  fillPoly(ctx, [x - s * 0.45, y, x, y + s * 0.18, x + s * 0.45, y, x, y - s * 0.18], "#8d9194", lw);
  line(ctx, [x - s * 0.12, y, x, y - s * 0.5, x + s * 0.12, y], "#3b3530", Math.max(0.8, s * 0.06));
  line(ctx, [x - s * 0.5, y - s * 0.58, x + s * 0.42, y - s * 0.4], "#d99a2b", Math.max(1, s * 0.09));
  fillPoly(ctx, [x - s * 0.5, y - s * 0.66, x - s * 0.6, y - s * 0.6, x - s * 0.58, y - s * 0.38, x - s * 0.48, y - s * 0.5], "#3b3530", 0);
  line(ctx, [x - s * 0.58, y - s * 0.4, x - s * 0.58, y], "#3b3530", Math.max(0.6, s * 0.03));
  blob(ctx, x + s * 0.32, y - s * 0.22, s * 0.11, s * 0.11, "#3b3530");
}
function derrick(ctx, x, y, s) {
  const base = y, top = y - s * 0.9, bw = s * 0.2, tw = s * 0.05, col = "#3b3530";
  line(ctx, [x - bw, base, x - tw, top], col, Math.max(0.9, s * 0.04));
  line(ctx, [x + bw, base, x + tw, top], col, Math.max(0.9, s * 0.04));
  ctx.beginPath();
  for (let k = 0; k < 4; k++) {
    const f0 = k / 4, f1 = (k + 1) / 4;
    const y0 = base + (top - base) * f0, y1 = base + (top - base) * f1;
    const h0 = bw + (tw - bw) * f0, h1 = bw + (tw - bw) * f1;
    ctx.moveTo(x - h0, y0); ctx.lineTo(x + h1, y1);
    ctx.moveTo(x + h0, y0); ctx.lineTo(x - h1, y1);
  }
  ctx.strokeStyle = col;
  ctx.lineWidth = Math.max(0.5, s * 0.02);
  ctx.stroke();
  blob(ctx, x, top - s * 0.04, s * 0.05, s * 0.05, "#e2493a");
}
function tank(ctx, x, y, r, hgt, c, lw) {
  fillPoly(ctx, [x - r, y - hgt, x, y - hgt, x, y, x - r, y], "#dfe3e5", 0);
  fillPoly(ctx, [x, y - hgt, x + r, y - hgt, x + r, y, x, y], "#9aa2a7", 0);
  blob(ctx, x, y, r, r * 0.35, "#9aa2a7");
  ctx.fillStyle = shade(c, 0);
  ctx.fillRect(x - r, y - hgt * 0.62, r * 2, hgt * 0.16);
  blob(ctx, x, y - hgt, r, r * 0.35, "#eef0f1");
  if (lw) {
    ctx.strokeStyle = INK;
    ctx.lineWidth = lw;
    ctx.strokeRect(x - r, y - hgt, r * 2, hgt);
  }
}

function oilTile(ctx, x, y, z, R, c) {
  const lw = lineWidth(z);
  groundFill(ctx, x, y, z, "#b6a685");
  speckle(ctx, x, y, z, R, 12, ["rgba(30,26,22,0.32)"], 0.1);
  // 管线
  ctx.strokeStyle = "#7d8286";
  ctx.lineWidth = Math.max(1, z * 0.03);
  ctx.beginPath();
  ctx.moveTo(x - z * 0.5, y - z * 0.1);
  ctx.lineTo(x + z * 0.1, y + z * 0.2);
  ctx.lineTo(x + z * 0.45, y + z * 0.02);
  ctx.moveTo(x + z * 0.1, y + z * 0.2);
  ctx.lineTo(x - z * 0.1, y + z * 0.6);
  ctx.stroke();
  const items = [
    { px: -0.12, py: -0.32, kind: "derrick" },
    { px: -0.5, py: -0.08, kind: "pump" },
    { px: 0.16, py: 0.22, kind: "pump" },
    { px: -0.12, py: 0.62, kind: "pump" },
    { px: 0.46, py: 0.0, kind: "tank" },
    { px: 0.62, py: 0.16, kind: "tank" },
  ];
  items.sort((p, q) => p.py - q.py);
  for (const it of items) {
    const bx = x + it.px * z, by = y + it.py * z;
    if (it.kind === "derrick") derrick(ctx, bx, by, z * 0.75);
    else if (it.kind === "pump") pumpjack(ctx, bx, by, z * 0.36, lw);
    else tank(ctx, bx, by, z * 0.1, z * 0.13, c, lw);
  }
}

function industrialBuilding(ctx, x, y, a, b, h, z, lw, kind, warm) {
  const wallL = warm ? "#c19777" : "#d0c7aa", wallR = warm ? "#967354" : "#a4997f";
  isoBox(ctx, x, y, a, h, wallL, wallR, lw, b);
  windows(ctx, x, y, a, h, "#526562");
  if (kind === "office") {
    flatRoof(ctx, x, y, a, h, "#738581", lw, b);
    flatRoof(ctx, x, y - z * 0.008, a * 0.65, h, "#a1ada0", 0, b * 0.65);
  } else if (kind === "warehouse") {
    gableRoof(ctx, x, y, a, b, h, z * 0.06, "#89978d", "#536b66", wallR, lw);
  } else {
    // 锯齿采光顶让生产车间与普通仓库有不同轮廓。
    flatRoof(ctx, x, y, a, h, "#687f79", lw, b);
    const ka = a * ISO, kb = b * ISO, rise = z * 0.042;
    for (let i = 0; i < 3; i++) {
      const t0 = i / 3, t1 = (i + 1) / 3;
      const x0 = x - a + a * t0, y0 = y - ka + ka * t0 - h;
      const x1 = x - a + a * t1, y1 = y - ka + ka * t1 - h;
      fillPoly(ctx, [x0,y0,x1,y1-rise,x1+b,y1-kb-rise,x0+b,y0-kb], "#819992", lw);
      fillPoly(ctx, [x1,y1,x1,y1-rise,x1+b,y1-kb-rise,x1+b,y1-kb], "#b1c7b8", lw);
    }
  }
  // 大门和侧窗沿墙面投影，不用醒目的黑色线框。
  fillPoly(ctx, [x + b*0.28,y - b*ISO*0.28,x + b*0.28,y - b*ISO*0.28-h*0.6,x + b*0.65,y - b*ISO*0.65-h*0.6,x + b*0.65,y - b*ISO*0.65], "#657468", 0);
}

function factoryTile(ctx, x, y, z, R) {
  const lw = lineWidth(z);
  groundFill(ctx, x, y, z, "#a3aa96");
  const paving = ctx.createLinearGradient(x - z, y - z, x + z, y + z);
  paving.addColorStop(0, "rgba(206,204,175,0.32)"); paving.addColorStop(1, "rgba(78,94,72,0.14)");
  ctx.fillStyle = paving; ctx.fillRect(x - z, y - z, z*2, z*2);
  speckle(ctx, x, y, z, R, 90, ["rgba(211,207,173,0.2)","rgba(89,103,78,0.14)"], 0.012);
  // 配套管线在建筑后面；交通道路不画进地块。
  line(ctx, [x-z*0.25,y-z*0.3,x+z*0.22,y-z*0.065,x+z*0.46,y-z*0.185], "#b6b7a1", z*0.02);
  const items = [
    {px:-0.34,py:-0.35,a:0.24,b:0.16,h:0.15,kind:"hall"},
    {px:0.22,py:-0.32,a:0.23,b:0.14,h:0.14,kind:"warehouse"},
    {px:-0.04,py:-0.5,kind:"chimneys"},
    {px:-0.53,py:0.04,a:0.13,b:0.08,h:0.2,kind:"office"},
    {px:0.52,py:-0.08,a:0.12,b:0.085,h:0.2,kind:"office"},
    {px:-0.04,py:0.02,a:0.27,b:0.19,h:0.16,kind:"hall"},
    {px:0.41,py:0.23,a:0.19,b:0.13,h:0.12,kind:"hall"},
    {px:-0.32,py:0.41,a:0.22,b:0.14,h:0.12,kind:"warehouse"},
    {px:0.03,py:0.56,a:0.21,b:0.16,h:0.13,kind:"hall"},
    {px:0.46,py:0.55,kind:"tanks"},
    {px:-0.62,py:0.35,kind:"crates"},
  ];
  items.sort((a,b)=>a.py-b.py);
  for (const it of items) {
    const bx=x+it.px*z,by=y+it.py*z;
    if (it.kind === "chimneys") {
      for (const off of [-0.055,0.055]) {
        const tx=bx+off*z,ty=by-Math.abs(off)*z*0.5,h=z*(off<0?0.31:0.25);
        isoBox(ctx,tx,ty,z*0.03,h,"#b88f70","#81644e",lw);
        flatRoof(ctx,tx,ty,z*0.03,h,"#635e4c",lw);
        line(ctx,[tx-z*0.03,ty-h*0.78,tx,ty-h*0.78+z*0.015,tx+z*0.03,ty-h*0.78],"#d3c7a1",z*0.012);
        for (let k=0;k<3;k++) blob(ctx,tx+z*(0.02+k*0.03),ty-h-z*(0.03+k*0.04),z*(0.026+k*0.009),z*(0.019+k*0.006),`rgba(219,218,199,${0.45-k*0.11})`);
      }
    } else if (it.kind === "tanks") {
      for (const off of [-0.07,0.08]) tank(ctx,bx+off*z,by-Math.abs(off)*z*0.5,z*0.07,z*0.16,[126,149,129],lw);
    } else if (it.kind === "crates") {
      for (let k=0;k<3;k++) isoBox(ctx,bx+(k%2)*z*0.055,by-Math.floor(k/2)*z*0.035,z*0.035,z*0.05,"#b9a77e","#8c825f",lw);
    } else industrialBuilding(ctx,bx,by,it.a*z,it.b*z,it.h*z,z,lw,it.kind,it.kind==="hall");
  }
}

function baseTile(ctx, x, y, z, R, c) {
  const lw = lineWidth(z);
  groundFill(ctx, x, y, z, "#b9a97d");
  speckle(ctx, x, y, z, R, 18, ["rgba(110,95,60,0.3)"], 0.03);
  // 操场
  fillPoly(ctx, [x - z * 0.42, y + z * 0.18, x, y + z * 0.39, x + z * 0.42, y + z * 0.18, x, y - z * 0.03], "#d8cba4", lw);
  // 铁丝网围栏
  ctx.save();
  ctx.setLineDash([z * 0.05, z * 0.04]);
  ctx.beginPath();
  for (let i = 0; i < 6; i++) {
    const a = ((60 * i - 30) * Math.PI) / 180;
    ctx.lineTo(x + Math.cos(a) * z * 0.9, y + Math.sin(a) * z * 0.9);
  }
  ctx.closePath();
  ctx.strokeStyle = "#5a5040";
  ctx.lineWidth = Math.max(0.8, z * 0.025);
  ctx.stroke();
  ctx.restore();
  const items = [
    { px: -0.36, py: -0.36, kind: "barracks" },
    { px: 0.12, py: -0.5, kind: "barracks" },
    { px: 0.56, py: -0.26, kind: "tower" },
    { px: -0.6, py: 0.08, kind: "tent" },
    { px: -0.44, py: 0.42, kind: "tent" },
    { px: 0.0, py: 0.18, kind: "flag" },
    { px: 0.5, py: 0.38, kind: "truck" },
  ];
  items.sort((p, q) => p.py - q.py);
  for (const it of items) {
    const bx = x + it.px * z, by = y + it.py * z;
    if (it.kind === "barracks") {
      isoBox(ctx, bx, by, z * 0.3, z * 0.08, "#d8cfb2", "#a99e7e", lw, z * 0.11);
      gableRoof(ctx, bx, by, z * 0.3, z * 0.11, z * 0.08, z * 0.06, "#6f7a45", "#4f5830", "#bcb08c", lw);
    } else if (it.kind === "tent") tent(ctx, bx, by, z * 0.1, z * 0.11, lw);
    else if (it.kind === "flag") flag(ctx, bx, by, z * 0.42, c, lw);
    else if (it.kind === "tower") {
      line(ctx, [bx - z * 0.05, by, bx - z * 0.03, by - z * 0.3, bx + z * 0.03, by - z * 0.3, bx + z * 0.05, by], "#5a4632", Math.max(0.8, z * 0.02));
      isoBox(ctx, bx, by - z * 0.3, z * 0.06, z * 0.07, "#b08a5c", "#7e6040", lw);
      hipRoof(ctx, bx, by - z * 0.3, z * 0.06, z * 0.07, z * 0.05, "#5f6a35", "#3c4322", lw);
    } else {
      isoBox(ctx, bx, by, z * 0.12, z * 0.06, "#6f7a45", "#4f5830", lw, z * 0.06);
      isoBox(ctx, bx + z * 0.03, by - z * 0.015, z * 0.04, z * 0.1, "#7d8850", "#58622f", lw, z * 0.06);
    }
  }
}

function plane(ctx, x, y, s, ang, c) {
  ctx.save();
  ctx.translate(x, y);
  ctx.rotate(ang);
  ctx.beginPath();
  const P = [[1, 0], [0.2, -0.12], [0.05, -0.9], [-0.2, -0.9], [-0.15, -0.12], [-0.75, -0.1], [-0.9, -0.4], [-1, -0.4], [-0.95, 0], [-1, 0.4], [-0.9, 0.4], [-0.75, 0.1], [-0.15, 0.12], [-0.2, 0.9], [0.05, 0.9], [0.2, 0.12]];
  P.forEach(([px, py], i) => (i ? ctx.lineTo(px * s, py * s) : ctx.moveTo(px * s, py * s)));
  ctx.closePath();
  ctx.fillStyle = shade(c, 0.05);
  ctx.fill();
  ctx.strokeStyle = "#f4f1ea";
  ctx.lineWidth = Math.max(0.6, s * 0.08);
  ctx.stroke();
  ctx.restore();
}

function airfieldTile(ctx, x, y, z, R, c) {
  const lw = lineWidth(z);
  groundFill(ctx, x, y, z, "#9cab64");
  // 修剪过的草坪条纹
  ctx.strokeStyle = "rgba(190,205,130,0.35)";
  ctx.lineWidth = z * 0.08;
  ctx.beginPath();
  for (let k = -6; k <= 6; k++) {
    ctx.moveTo(x - z * 1.2, y + k * 0.24 * z + z * 0.6);
    ctx.lineTo(x + z * 1.2, y + k * 0.24 * z - z * 0.6);
  }
  ctx.stroke();
  // 跑道：沿 E2 方向斜穿整格
  const ux = 0.894, uy = -0.447, vx = 0.894, vy = 0.447;
  const cx = x + z * 0.08, cy = y + z * 0.22, L = z * 1.3, W = z * 0.13;
  fillPoly(ctx, [cx - ux * L - vx * W, cy - uy * L - vy * W, cx + ux * L - vx * W, cy + uy * L - vy * W, cx + ux * L + vx * W, cy + uy * L + vy * W, cx - ux * L + vx * W, cy - uy * L + vy * W], "#55595c", 0);
  ctx.save();
  ctx.setLineDash([z * 0.09, z * 0.07]);
  line(ctx, [cx - ux * L, cy - uy * L, cx + ux * L, cy + uy * L], "#f2efe6", Math.max(0.7, z * 0.02));
  ctx.restore();
  for (const sgn of [-1, 1]) {
    line(ctx, [cx - ux * L + sgn * vx * W * 0.85, cy - uy * L + sgn * vy * W * 0.85, cx + ux * L + sgn * vx * W * 0.85, cy + uy * L + sgn * vy * W * 0.85], "rgba(242,239,230,0.7)", Math.max(0.5, z * 0.012));
  }
  // 停机坪（后方）
  const ax = x - z * 0.22, ay = y - z * 0.28;
  fillPoly(ctx, [ax - z * 0.38, ay, ax, ay + z * 0.19, ax + z * 0.38, ay, ax, ay - z * 0.19], "#7c8083", 0);
  // 机库（半圆拱）× 2、塔台、飞机、风向袋
  for (const [hx, hy] of [[x - z * 0.48, y - z * 0.4], [x - z * 0.16, y - z * 0.56]]) {
    const hr = z * 0.11;
    fillPoly(ctx, [hx + hr, hy, hx + hr + z * 0.13, hy - z * 0.065, hx + z * 0.13, hy - hr - z * 0.065, hx, hy - hr], "#878e94", lw);
    ctx.beginPath();
    ctx.moveTo(hx - hr, hy);
    ctx.quadraticCurveTo(hx - hr, hy - hr * 1.35, hx, hy - hr);
    ctx.quadraticCurveTo(hx + hr, hy - hr, hx + hr, hy);
    ctx.closePath();
    ctx.fillStyle = "#c3c9cd";
    ctx.fill();
    if (lw) {
      ctx.strokeStyle = INK;
      ctx.lineWidth = lw;
      ctx.stroke();
    }
    fillPoly(ctx, [hx - hr * 0.5, hy, hx - hr * 0.5, hy - hr * 0.55, hx + hr * 0.5, hy - hr * 0.55, hx + hr * 0.5, hy], "#3a3f43", 0);
  }
  const tx = x + z * 0.24, ty = y - z * 0.36;
  isoBox(ctx, tx, ty, z * 0.045, z * 0.32, "#e5ddcb", "#b3a98f", lw);
  isoBox(ctx, tx, ty - z * 0.32, z * 0.075, z * 0.07, "#9fd0e0", "#6aa2b6", lw);
  flatRoof(ctx, tx, ty - z * 0.32, z * 0.075, z * 0.07, "#4f5458", lw);
  plane(ctx, ax - z * 0.12, ay + z * 0.0, z * 0.12, -0.46, c);
  plane(ctx, cx + ux * z * 0.35, cy + uy * z * 0.35, z * 0.13, -0.46, c);
  const wx = x + z * 0.55, wy = y + z * 0.45;
  line(ctx, [wx, wy, wx, wy - z * 0.24], "#3a2e24", Math.max(0.8, z * 0.02));
  fillPoly(ctx, [wx, wy - z * 0.24, wx + z * 0.13, wy - z * 0.21, wx + z * 0.13, wy - z * 0.18, wx, wy - z * 0.16], "#ec7a2e", lw);
}

function ship(ctx, x, y, z, ang, c, lw) {
  ctx.save();
  ctx.translate(x, y);
  ctx.rotate(ang);
  const L = z * 0.42, W = z * 0.1;
  blob(ctx, z * 0.02, z * 0.03, L * 1.05, W * 1.3, "rgba(10,20,30,0.3)");
  fillPoly(ctx, [-L, -W, L * 0.75, -W, L, 0, L * 0.75, W, -L, W], "#4b525a", lw);
  fillPoly(ctx, [-L * 0.85, -W * 0.6, L * 0.68, -W * 0.6, L * 0.82, 0, L * 0.68, W * 0.6, -L * 0.85, W * 0.6], "#b9b2a2", 0);
  fillPoly(ctx, [-L * 0.45, -W * 0.5, -L * 0.05, -W * 0.5, -L * 0.05, W * 0.5, -L * 0.45, W * 0.5], "#eeeae0", lw);
  blob(ctx, -L * 0.25, 0, W * 0.32, W * 0.32, shade(c, 0));
  ctx.restore();
}

function portTile(ctx, x, y, z, R, c, info) {
  const lw = lineWidth(z);
  const water = [];
  for (let d = 0; d < 6; d++) if (info && info.coast & (1 << d)) water.push(d);
  if (!water.length) water.push(5);
  groundFill(ctx, x, y, z, "#c7bba0");
  // 临水的边砌成石头码头
  const edgeCorners = [[0, 1], [5, 0], [4, 5], [3, 4], [2, 3], [1, 2]];
  const corner = (i, r) => {
    const a = ((60 * i - 30) * Math.PI) / 180;
    return [x + Math.cos(a) * r, y + Math.sin(a) * r];
  };
  for (const d of water) {
    const [i0, i1] = edgeCorners[d];
    const [p0, p1, q1, q0] = [corner(i0, z), corner(i1, z), corner(i1, z * 0.86), corner(i0, z * 0.86)];
    fillPoly(ctx, [p0[0], p0[1], p1[0], p1[1], q1[0], q1[1], q0[0], q0[1]], "#9a9184", 0);
    line(ctx, [q0[0], q0[1], q1[0], q1[1]], "#d8d0c2", Math.max(0.6, z * 0.02));
  }
  // 码头上的货箱、仓库、起重机（离水近的放起重机，远的放仓库）
  const ang = water.map((d) => EDGE_ANGLE[d]);
  const nx = ang.reduce((s, a) => s + Math.cos(a), 0), ny = ang.reduce((s, a) => s + Math.sin(a), 0);
  const nl = Math.hypot(nx, ny) || 1;
  const n = [nx / nl, ny / nl];
  const items = [];
  for (const [px, py] of lattice(0.42, 0.76)) {
    const toward = px * n[0] + py * n[1];
    if (toward > 0.45) continue;
    if (toward > 0.15) items.push({ px, py, kind: R() < 0.5 ? "crane" : "crates" });
    else items.push({ px, py, kind: R() < 0.75 ? "warehouse" : "crates" });
  }
  items.sort((p, q) => p.py - q.py || p.px - q.px);
  const crateCols = ["#b5543a", "#3f6f95", "#d8a23a", "#5b8a4a"];
  for (const it of items) {
    const bx = x + it.px * z, by = y + it.py * z;
    if (it.kind === "warehouse") {
      const roof = ["#778b8a", "#4c6466"];
      isoBox(ctx, bx, by, z * 0.14, z * 0.1, "#e2d6bd", "#ad9d7f", lw, z * 0.1);
      gableRoof(ctx, bx, by, z * 0.14, z * 0.1, z * 0.1, z * 0.06, roof[0], roof[1], "#c9bb9b", lw);
    } else if (it.kind === "crates") {
      for (let k = 0; k < 3; k++) {
        const col = crateCols[Math.floor(R() * crateCols.length)];
        isoBox(ctx, bx + (k - 1) * z * 0.06, by + (k % 2) * z * 0.03, z * 0.035, z * 0.05, col, shade(hexRgb(col), -0.3), lw);
      }
    } else {
      const col = "#e3a12c", w = Math.max(1, z * 0.025);
      line(ctx, [bx - z * 0.05, by, bx - z * 0.03, by - z * 0.3], col, w);
      line(ctx, [bx + z * 0.05, by + z * 0.02, bx + z * 0.03, by - z * 0.28], col, w);
      line(ctx, [bx - z * 0.08, by - z * 0.3, bx + n[0] * z * 0.32, by - z * 0.34 + n[1] * z * 0.12], col, w);
      line(ctx, [bx + n[0] * z * 0.26, by - z * 0.33 + n[1] * z * 0.1, bx + n[0] * z * 0.26, by - z * 0.16 + n[1] * z * 0.1], "#3a2e24", Math.max(0.5, z * 0.01));
    }
  }
  // 栈桥和船伸到相邻的水面上，要在六边形裁剪之外画
  return () => {
    for (const d of water.slice(0, 2)) {
    const a = EDGE_ANGLE[d];
    const ex = x + Math.cos(a) * z * 0.86, ey = y + Math.sin(a) * z * 0.86;
    const tx = -Math.sin(a), ty = Math.cos(a);
    // 栈桥
    const L = z * 0.45, W = z * 0.07;
    const px = ex + tx * z * 0.18, py = ey + ty * z * 0.18;
    fillPoly(ctx, [px - tx * W, py - ty * W, px - tx * W + Math.cos(a) * L, py - ty * W + Math.sin(a) * L, px + tx * W + Math.cos(a) * L, py + ty * W + Math.sin(a) * L, px + tx * W, py + ty * W], "#b8895a", lw);
    ship(ctx, ex + Math.cos(a) * z * 0.3 - tx * z * 0.12, ey + Math.sin(a) * z * 0.3 - ty * z * 0.12, z, a + Math.PI / 2, c, lw);
    }
  };
}

const TILE_ART = {
  farm: farmTile,
  mine: mineTile,
  lumber: lumberTile,
  plantation: plantationTile,
  oil: oilTile,
  factory: factoryTile,
  base: baseTile,
  airfield: airfieldTile,
  port: portTile,
};

// 设施铺满整个六边形；港口栈桥保留在相邻水面上。
const ART_VARIANTS = 6;
const ART_KEY = {
  mine: (c, info) => `${info?.color}:${info?.resource}`,
  plantation: (c, info) => `${info?.resource}`,
  oil: (c) => `${c}`,
  base: (c) => `${c}`,
  airfield: (c) => `${c}`,
  port: (c, info) => `${c}:${info?.coast}`,
};

export function drawFacilityTile(ctx, icon, x, y, z, c, info, seed) {
  const draw = TILE_ART[icon];
  if (!draw) return;
  if (z < (G.snap?.stress ? 28 : LOD.tileArt)) {
    ctx.save(); hexClip(ctx, x, y, z);
    groundFill(ctx, x, y, z, icon === "farm" ? "#91a346" : TILE_GROUND[icon]);
    ctx.restore();
    return;
  }
  // 同一种设施只画 ART_VARIANTS 种样子，按格子编号挑一种；只有画法真正用到的参数（国家色、
  // 资源、地形色、临水方向）才进缓存键。按格子各画一张的话，满图时贴图缓存装不下、每帧重画
  const variant = hash32(seed) % ART_VARIANTS;
  const key = `facility:${icon}:${variant}:${ART_KEY[icon]?.(c, info) ?? ""}`;
  tileArtwork(ctx, key, x, y, z, (cx, px, py, s) => {
    cx.save(); hexClip(cx, px, py, s);
    cx.lineCap = "round"; cx.lineJoin = "round";
    const outside = draw(cx, px, py, s, rand(hash32(variant * 7919 + icon.length * 104729)), c, info);
    cx.restore();
    if (outside) { cx.save(); cx.lineCap = "round"; cx.lineJoin = "round"; outside(); cx.restore(); }
  }, ["farm", "mine", "factory"].includes(icon) ? 128 : ART_RADIUS);
}

// ---------------------------------------------------------------- 城市

function drawCityPlate(ctx, x, y, z, city, c) {
  ctx.save();
  const fs = Math.round(Math.max(10, Math.min(14, 9 + z * 0.12)));
  const ph = fs + 12;
  ctx.font = `600 ${fs}px ${FONT}`;
  const cityLabel = `${city.name} · L${city.level ?? 1}`;
  const tw = ctx.measureText(cityLabel).width;
  const pw = tw + ph * 2 + 16;
  const px = x - pw / 2, py = y - Math.max(z * 1.05, 12) - ph;
  ctx.shadowColor = "rgba(10,17,22,0.45)"; ctx.shadowBlur = 7; ctx.shadowOffsetY = 2;
  ctx.beginPath(); roundRect(ctx, px, py, pw, ph, 5);
  const bg = ctx.createLinearGradient(0, py, 0, py + ph);
  bg.addColorStop(0, "#283a40"); bg.addColorStop(1, "#15242b");
  ctx.fillStyle = bg; ctx.fill();
  ctx.shadowBlur = 0; ctx.shadowOffsetY = 0;
  ctx.strokeStyle = "#bda578"; ctx.lineWidth = 0.8; ctx.stroke();
  const bx = px + ph / 2, by = py + ph / 2;
  ctx.beginPath(); ctx.arc(bx, by, ph * 0.34, 0, Math.PI * 2);
  ctx.fillStyle = shade(c, -0.1); ctx.fill();
  ctx.strokeStyle = "#dbc596"; ctx.lineWidth = 0.8; ctx.stroke();
  ctx.fillStyle = "#f3e5c8"; ctx.textAlign = "center"; ctx.textBaseline = "middle";
  if (city.capital) { drawStar(ctx, bx, by, fs * 0.34); ctx.fill(); }
  else fillPoly(ctx, [bx, by - fs * 0.24, bx + fs * 0.24, by, bx, by + fs * 0.24, bx - fs * 0.24, by], "#f3e5c8", 0);
  ctx.font = `600 ${fs}px ${FONT}`; ctx.textAlign = "left";
  ctx.fillText(cityLabel, px + ph + 3, by);
  // 动荡（稳定度低于门槛）：名牌下方挂一块红牌，周围有城市独立时会一起独立
  if ((city.stability ?? 100) < (G.rules.stability_unrest ?? 30)) {
    const tag = `动荡 ${city.stability}`, tfs = Math.max(9, fs - 2);
    ctx.font = `700 ${tfs}px ${FONT}`;
    const tw2 = ctx.measureText(tag).width + 10, ty = py + ph + 2;
    ctx.beginPath(); roundRect(ctx, x - tw2 / 2, ty, tw2, tfs + 6, 3);
    ctx.fillStyle = "#9b2f24"; ctx.fill(); ctx.strokeStyle = "#f2b8a6"; ctx.lineWidth = 0.8; ctx.stroke();
    ctx.fillStyle = "#fbe9df"; ctx.textAlign = "center"; ctx.textBaseline = "middle";
    ctx.fillText(tag, x, ty + (tfs + 6) / 2);
    ctx.font = `600 ${fs}px ${FONT}`; ctx.textBaseline = "middle";
  }
  if ((city.hp ?? 200) < 200) {
    ctx.fillStyle = "#503b35"; ctx.fillRect(px + 4, py - 5, pw - 8, 3);
    ctx.fillStyle = "#dba283"; ctx.fillRect(px + 4, py - 5, (pw - 8) * city.hp / 200, 3);
  }
  line(ctx, [px + pw - ph, py + 6, px + pw - ph, py + ph - 6], "rgba(217,192,147,0.3)", 0.7);
  ctx.textAlign = "center"; ctx.fillStyle = "#d8bf89";
  ctx.fillText(String(city.population), px + pw - ph / 2, by);
  if (z >= 24) {
    const w = pw - 8, by2 = py + ph + 3;
    ctx.fillStyle = "rgba(14,25,28,0.75)"; ctx.fillRect(px + 4, by2, w, 3);
    ctx.fillStyle = "#9cb47d"; ctx.fillRect(px + 4, by2, w * Math.min(1, city.food / Math.max(1, city.food_to_grow)), 3);
    if (city.queue.length) {
      const [kind, key] = city.queue[0].split(":");
      const def = kind === "unit" ? G.rules.units[key] : G.rules.buildings[key];
      if (def) {
        ctx.font = `500 ${fs - 2}px ${FONT}`;
        const label = def.name, sw = Math.max(pw - 18, ctx.measureText(label).width + 14);
        const ly = by2 + 6;
        ctx.beginPath(); roundRect(ctx, x - sw / 2, ly, sw, fs + 5, 3);
        ctx.fillStyle = "rgba(20,33,38,0.92)"; ctx.fill();
        ctx.fillStyle = "#bda578";
        ctx.fillRect(x - sw / 2 + 2, ly + fs + 3, (sw - 4) * Math.min(1, city.production / Math.max(1, def.cost)), 2);
        ctx.fillStyle = "#ece1c9"; ctx.fillText(label, x, ly + (fs + 3) / 2);
      }
    }
  }
  ctx.restore();
}

// 远景的城市标记（近景用 drawUnder 的景观）
function drawCityMarkers(ctx, v) {
  const z = v.z;
  if (z >= (G.snap.stress ? 28 : LOD.tileArt)) return;
  for (const [,city] of visibleEntries(G.cityAt,v)) {
    const c = G.nations.get(city.nation).rgb;
    const [wx, wy] = center(city.tile);
    eachCopy(v, wx, wy, 20, (x, y) => {
      const r = Math.max(2.5, Math.min(5, z * 0.9));
      ctx.beginPath();
      if (city.capital) drawStar(ctx, x, y, r * 1.5);
      else ctx.arc(x, y, r, 0, Math.PI * 2);
      ctx.fillStyle = shade(c, -0.1);
      ctx.fill();
      ctx.lineWidth = 1.2;
      ctx.strokeStyle = "#fbf6e8";
      ctx.stroke();
    });
  }
}

function drawCityPlates(ctx, v) {
  const z = v.z;
  if (z < (G.snap.stress ? 16 : LOD.cityPlate) || (G.political && z < POLITICAL_LABELS_BELOW)) return;
  for (const [,city] of visibleEntries(G.cityAt,v)) {
    const c = G.nations.get(city.nation).rgb;
    const [wx, wy] = center(city.tile);
    eachCopy(v, wx, wy, 120, (x, y) => drawCityPlate(ctx, x, y, z, city, c));
  }
}

// ---------------------------------------------------------------- 兵种符号（北约军标风格）

function drawSymbol(ctx, icon, x, y, w, h, ink) {
  // 在 (x, y) 为中心、w × h 的符号框里画兵种符号
  ctx.save();
  ctx.strokeStyle = ink;
  ctx.fillStyle = ink;
  ctx.lineWidth = Math.max(1, w * 0.07);
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  const l = x - w / 2, r = x + w / 2, t = y - h / 2, b = y + h / 2;
  ctx.beginPath();
  switch (icon) {
    case "infantry":
      ctx.moveTo(l, t); ctx.lineTo(r, b); ctx.moveTo(r, t); ctx.lineTo(l, b);
      break;
    case "motorized":
      ctx.moveTo(l, t); ctx.lineTo(r, b); ctx.moveTo(r, t); ctx.lineTo(l, b);
      ctx.moveTo(x, t); ctx.lineTo(x, b);
      break;
    case "armor":
      ctx.ellipse(x, y, w * 0.34, h * 0.26, 0, 0, Math.PI * 2);
      break;
    case "artillery":
      ctx.arc(x, y, Math.min(w, h) * 0.17, 0, Math.PI * 2);
      ctx.fill();
      break;
    case "anti_air": {
      // 防空：框底的穹顶弧线 + 炮兵圆点
      const s = Math.min(w * 0.42, h * 0.62);
      ctx.moveTo(x - s, b - h * 0.06); ctx.quadraticCurveTo(x, b - h * 0.06 - s * 1.1, x + s, b - h * 0.06);
      ctx.stroke(); ctx.beginPath();
      ctx.arc(x, y - h * 0.12, Math.min(w, h) * 0.13, 0, Math.PI * 2);
      ctx.fill();
      break;
    }
    case "recon":
      ctx.moveTo(l, b); ctx.lineTo(r, t);
      break;
    case "engineer": {
      // 横放的 "E"：一条横线 + 三根短竖
      const yy = y - h * 0.12, d = h * 0.3;
      ctx.moveTo(x - w * 0.32, yy + d); ctx.lineTo(x - w * 0.32, yy); ctx.lineTo(x + w * 0.32, yy); ctx.lineTo(x + w * 0.32, yy + d);
      ctx.moveTo(x, yy); ctx.lineTo(x, yy + d);
      break;
    }
    case "general": {
      // 将领：五角星
      const r = Math.min(w, h) * 0.46;
      for (let k = 0; k < 10; k++) {
        const a = -Math.PI / 2 + (k * Math.PI) / 5, rr = k % 2 ? r * 0.42 : r;
        const px = x + Math.cos(a) * rr, py = y + Math.sin(a) * rr;
        if (k) ctx.lineTo(px, py); else ctx.moveTo(px, py);
      }
      ctx.closePath(); ctx.fill();
      break;
    }
    case "settler": {
      const s = Math.min(w, h) * 0.36;
      ctx.moveTo(x - s, y + s * 0.8); ctx.lineTo(x - s, y - s * 0.1); ctx.lineTo(x, y - s * 0.9);
      ctx.lineTo(x + s, y - s * 0.1); ctx.lineTo(x + s, y + s * 0.8); ctx.closePath();
      break;
    }
    case "fighter":
    case "bomber": {
      const s = Math.min(w, h) * 0.42;
      ctx.moveTo(x, y - s); ctx.lineTo(x, y + s); // 机身
      ctx.moveTo(x - s * 1.1, y + s * 0.1); ctx.lineTo(x, y - s * 0.25); ctx.lineTo(x + s * 1.1, y + s * 0.1); // 机翼
      ctx.moveTo(x - s * 0.45, y + s * 0.85); ctx.lineTo(x, y + s * 0.65); ctx.lineTo(x + s * 0.45, y + s * 0.85); // 尾翼
      if (icon === "bomber") { ctx.moveTo(x - s * 0.6, y - s * 0.02); ctx.lineTo(x - s * 0.6, y + s * 0.3); ctx.moveTo(x + s * 0.6, y - s * 0.02); ctx.lineTo(x + s * 0.6, y + s * 0.3); }
      break;
    }
    case "destroyer":
    case "cruiser":
    case "battleship": {
      // 船身 + 桅杆；巡洋多一座矮舰楼，战列是长舰楼
      const s = w * 0.4;
      ctx.moveTo(x - s, y + h * 0.05); ctx.lineTo(x + s, y + h * 0.05); ctx.lineTo(x + s * 0.7, y + h * 0.3); ctx.lineTo(x - s * 0.75, y + h * 0.3); ctx.closePath();
      ctx.moveTo(x - s * 0.1, y + h * 0.05); ctx.lineTo(x - s * 0.1, y - h * 0.3);
      if (icon === "cruiser") { ctx.moveTo(x - s * 0.4, y + h * 0.05); ctx.lineTo(x - s * 0.4, y - h * 0.05); ctx.lineTo(x + s * 0.15, y - h * 0.05); ctx.lineTo(x + s * 0.15, y + h * 0.05); }
      if (icon === "battleship") { ctx.moveTo(x - s * 0.55, y + h * 0.05); ctx.lineTo(x - s * 0.55, y - h * 0.1); ctx.lineTo(x + s * 0.4, y - h * 0.1); ctx.lineTo(x + s * 0.4, y + h * 0.05); }
      break;
    }
    case "transport": {
      // 运输船：平底船身 + 甲板上的货箱
      const s = w * 0.42;
      ctx.moveTo(x - s, y + h * 0.05); ctx.lineTo(x + s, y + h * 0.05); ctx.lineTo(x + s * 0.85, y + h * 0.3); ctx.lineTo(x - s * 0.85, y + h * 0.3); ctx.closePath();
      ctx.rect(x - s * 0.6, y - h * 0.2, s * 0.5, h * 0.25); ctx.rect(x + s * 0.05, y - h * 0.2, s * 0.5, h * 0.25);
      break;
    }
    case "carrier": {
      // 平直的飞行甲板 + 右舷舰岛
      const s = w * 0.42;
      ctx.moveTo(x - s, y + h * 0.02); ctx.lineTo(x + s, y + h * 0.02); ctx.lineTo(x + s * 0.8, y + h * 0.3); ctx.lineTo(x - s * 0.8, y + h * 0.3); ctx.closePath();
      ctx.moveTo(x + s * 0.35, y + h * 0.02); ctx.lineTo(x + s * 0.35, y - h * 0.22); ctx.lineTo(x + s * 0.6, y - h * 0.22); ctx.lineTo(x + s * 0.6, y + h * 0.02);
      ctx.moveTo(x - s * 0.7, y - h * 0.08); ctx.lineTo(x + s * 0.1, y - h * 0.08);
      break;
    }
    default:
      ctx.font = `700 ${Math.round(h * 0.7)}px ${FONT}`;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(icon.slice(0, 1), x, y);
  }
  ctx.stroke();
  ctx.restore();
}

// ---------------------------------------------------------------- 单位牌

function hpColor(hp) {
  return hp > 66 ? "#5fbf5a" : hp > 33 ? "#e0b84a" : "#d0573f";
}

// 地表单位牌（矩形）。full = 完整牌面（战斗力、数量、血条）
function drawCard(ctx, x, y, w, h, u, count, c, full) {
  const def = unitDef(u);
  const r = Math.max(2, w * 0.08);
  ctx.save();
  ctx.shadowColor = "rgba(0,0,0,0.45)";
  ctx.shadowBlur = Math.max(2, w * 0.08);
  ctx.shadowOffsetY = Math.max(1, w * 0.03);
  ctx.beginPath();
  roundRect(ctx, x - w / 2, y - h / 2, w, h, r);
  ctx.fillStyle = shade(c, -0.08);
  ctx.fill();
  ctx.restore();
  ctx.beginPath();
  roundRect(ctx, x - w / 2, y - h / 2, w, h, r);
  // 深色外描边：浅色地形（沙漠、平原）上兵牌也有清楚的轮廓
  ctx.lineWidth = Math.max(1, w * 0.035) + 2;
  ctx.strokeStyle = "rgba(16,13,9,0.8)";
  ctx.stroke();
  ctx.lineWidth = Math.max(1, w * 0.035);
  ctx.strokeStyle = u.fortified ? "#1a140a" : "rgba(255,248,230,0.7)";
  ctx.stroke();
  // 符号框：平民用圆角框，军事用直角框
  const bw = w * (full ? 0.5 : 0.62), bh = h * (full ? 0.42 : 0.56);
  const by = full ? y - h * 0.04 : y;
  ctx.beginPath();
  if (def.category === "civilian") roundRect(ctx, x - bw / 2, by - bh / 2, bw, bh, bh * 0.35);
  else ctx.rect(x - bw / 2, by - bh / 2, bw, bh);
  // 完整牌面用白底符号框；小牌不铺白底，整张都是国家色，远一点也认得出是谁的部队
  const ink = full ? "#1f1a12" : inkOn(c);
  if (full) {
    ctx.fillStyle = "#f7f1e3";
    ctx.fill();
  }
  ctx.lineWidth = Math.max(1, w * (full ? 0.03 : 0.04));
  ctx.strokeStyle = ink;
  ctx.stroke();
  drawSymbol(ctx, def.icon, x, by, bw * 0.78, bh * 0.72, ink);
  const fs = Math.max(8, Math.round(h * 0.24));
  ctx.textBaseline = "middle";
  if (full) {
    ctx.font = `700 ${fs}px ${FONT}`;
    ctx.fillStyle = "#fbf6e8";
    ctx.textAlign = "left";
    const ranged = def.ranged_strength || def.anti_air || def.torpedo;
    const str = u.rank ? `★${u.rank}` : ranged ? `${u.combat_strength ?? def.strength}/${u.ranged_strength ?? ranged}` : def.strength ? String(u.combat_strength ?? def.strength) : def.short;
    if (u.rank) ctx.fillStyle = "#ffd479";
    ctx.fillText(str, x - w / 2 + w * 0.06, y - h / 2 + fs * 0.62);
    ctx.font = `600 ${Math.max(8, fs * .75)}px ${FONT}`;
    ctx.fillStyle = "#ffe2a0"; ctx.textAlign = "left";
    if (!u.rank) ctx.fillText(`L${u.level ?? 0}`, x - w / 2 + w * .07, y + h * .23);
    // 军团成员：右下角标军团番号（第 3 军团 → 3军）
    const corps = u.corps != null ? G.units.get(u.corps) : null;
    if (corps?.title) {
      ctx.textAlign = "right"; ctx.fillStyle = "#ffd479";
      ctx.fillText(`${corps.title.replace(/[^0-9]/g, "")}军`, x + w / 2 - w * .06, y + h * .23);
    }
    // 将领：牌下方写名字和番号
    if (u.rank && u.name) {
      ctx.save();
      ctx.font = `700 ${Math.max(9, fs * .8)}px ${FONT}`; ctx.textAlign = "center";
      ctx.lineJoin = "round"; ctx.lineWidth = Math.max(2, fs * .28); ctx.strokeStyle = "rgba(20,16,10,.85)";
      const label = `${u.title} · ${u.name}`;
      ctx.strokeText(label, x, y + h / 2 + fs * .75);
      ctx.fillStyle = "#ffe7b0"; ctx.fillText(label, x, y + h / 2 + fs * .75);
      ctx.restore();
    }
    // 血条
    const hbw = w * 0.84, hbh = Math.max(2, h * 0.08);
    const hx = x - hbw / 2, hy = y + h / 2 - hbh - h * 0.07;
    ctx.fillStyle = "rgba(0,0,0,0.45)";
    ctx.fillRect(hx, hy, hbw, hbh);
    const pct = (100 * u.hp) / (u.max_hp ?? 100);
    ctx.fillStyle = hpColor(pct);
    ctx.fillRect(hx, hy, (hbw * pct) / 100, hbh);
    // 施工 / 驻守标记
    if (u.task) {
      ctx.font = `600 ${Math.round(fs * 0.85)}px ${FONT}`;
      ctx.textAlign = "right";
      ctx.fillStyle = "#ffd479";
      ctx.fillText(`建${u.task.turns_left}`, x + w / 2 - w * 0.05, y + h * 0.12);
    }
  }
  if (count > 1) {
    const br = Math.max(6, fs * 0.62);
    const bx = x + w / 2 - (full ? br + w * 0.04 : br * 0.3);
    const byy = y - h / 2 + (full ? br + h * 0.04 : br * 0.3);
    ctx.beginPath();
    ctx.arc(bx, byy, br, 0, Math.PI * 2);
    ctx.fillStyle = "#1a140a";
    ctx.fill();
    ctx.lineWidth = 1;
    ctx.strokeStyle = "rgba(255,248,230,0.8)";
    ctx.stroke();
    ctx.fillStyle = "#fbf6e8";
    ctx.textAlign = "center";
    ctx.font = `700 ${Math.round(br * 1.1)}px ${FONT}`;
    ctx.fillText(String(count), bx, byy + 0.5);
  }
}

// 航空单位牌（圆角菱形，悬浮 + 投影）
function drawAirCard(ctx, x, y, s, u, count, c) {
  const def = unitDef(u);
  const diamond = (ox, oy) => {
    // 旋转 45° 的圆角方块
    const a = s * 1.32;
    ctx.save();
    ctx.translate(x + ox, y + oy);
    ctx.rotate(Math.PI / 4);
    ctx.beginPath();
    roundRect(ctx, -a / 2, -a / 2, a, a, a * 0.18);
    ctx.restore();
  };
  diamond(s * 0.25, s * 0.45); // 投影：离地面有一段距离
  ctx.fillStyle = "rgba(0,0,0,0.28)";
  ctx.fill();
  diamond(0, 0);
  ctx.fillStyle = shade(c, 0.05);
  ctx.fill();
  ctx.lineWidth = Math.max(1, s * 0.06);
  ctx.strokeStyle = "rgba(255,248,230,0.85)";
  ctx.stroke();
  ctx.beginPath();
  ctx.arc(x, y, s * 0.48, 0, Math.PI * 2);
  ctx.fillStyle = "#f7f1e3";
  ctx.fill();
  drawSymbol(ctx, def.icon, x, y, s * 0.72, s * 0.72, "#1f1a12");
  if (s >= 10) {
    ctx.font = `600 ${Math.max(8, s*.38)}px ${FONT}`; ctx.textAlign='center';ctx.textBaseline='middle';
    ctx.strokeStyle='#24312b';ctx.lineWidth=3;ctx.strokeText(`L${u.level??0}`,x,y+s*.85);
    ctx.fillStyle='#f2d79e';ctx.fillText(`L${u.level??0}`,x,y+s*.85);
  }
  if (count > 1) {
    const fs = Math.max(8, Math.round(s * 0.45));
    ctx.font = `700 ${fs}px ${FONT}`;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.lineWidth = 3;
    ctx.strokeStyle = "#1a140a";
    ctx.strokeText(`×${count}`, x + s * 0.95, y - s * 0.75);
    ctx.fillStyle = "#fbf6e8";
    ctx.fillText(`×${count}`, x + s * 0.95, y - s * 0.75);
  }
}

function drawStack(ctx, x, y, z, st) {
  const w = Math.max(24, Math.min(88, SQRT3 * z * 0.7));
  const h = w * 0.76;
  const full = w >= LOD.fullCard;
  if (st.surface.length) {
    const top = st.surface[0];
    const c = G.nations.get(top.nation).rgb;
    const behind = Math.min(2, st.surface.length - 1);
    const off = Math.max(2, w * 0.07);
    for (let i = behind; i >= 1; i--) {
      // 下面的牌只露出边
      ctx.beginPath();
      roundRect(ctx, x - w / 2 + off * i, y - h / 2 - off * i, w, h, Math.max(2, w * 0.08));
      ctx.fillStyle = shade(c, -0.3 + 0.08 * i);
      ctx.fill();
      ctx.lineWidth = 1;
      ctx.strokeStyle = "rgba(255,248,230,0.45)";
      ctx.stroke();
    }
    drawCard(ctx, x, y, w, h, top, st.surface.length, c, full);
  }
  if (st.air.length) {
    const top = st.air[0];
    const c = G.nations.get(top.nation).rgb;
    const s = w * 0.36;
    const ax = st.surface.length ? x + w * 0.5 : x;
    const ay = st.surface.length ? y - h * 0.62 : y - h * 0.2;
    drawAirCard(ctx, ax, ay, s, top, st.air.length, c);
  }
}

function drawArmyDots(ctx, v) {
  // 远景：不画牌，城外的部队画成小菱形点（城里的部队由城市标记代表）
  const s = Math.max(2.5, Math.min(4.5, v.z * 0.7));
  for (const [t, st] of visibleEntries(G.stacks,v)) {
    if (G.cityAt.has(t)) continue;
    const u = st.surface[0] || st.air[0];
    const c = G.nations.get(u.nation).rgb;
    const [wx, wy] = center(t);
    eachCopy(v, wx, wy, 10, (x, y) => {
      ctx.beginPath();
      ctx.moveTo(x, y - s); ctx.lineTo(x + s, y); ctx.lineTo(x, y + s); ctx.lineTo(x - s, y); ctx.closePath();
      ctx.fillStyle = shade(c, -0.15);
      ctx.fill();
      ctx.lineWidth = 1;
      ctx.strokeStyle = "#fbf6e8";
      ctx.stroke();
    });
  }
}

// 国家色上用深色还是浅色字 / 符号（金黄、浅绿这类亮色底上白字看不清）
function inkOn([r, g, b]) {
  return 0.299 * r + 0.587 * g + 0.114 * b > 150 ? "#1f1a12" : "#fbf6e8";
}

// 远景、中景的部队：固定屏幕大小的小兵牌（国家色底 + 深色描边），不随缩放缩成看不见的点。
// 有军事单位：单支画兵种符号、多支画数量；只有平民单位（开拓者、工人）的格子画成小圆点
function drawToken(ctx, x, y, z, units) {
  const military = units.filter((u) => unitDef(u)?.category !== "civilian");
  const top = military[0] ?? units[0];
  const c = G.nations.get(top.nation)?.rgb;
  if (!c) return;
  ctx.save();
  if (!military.length) {
    ctx.beginPath();
    ctx.arc(x, y, 5, 0, Math.PI * 2);
    ctx.fillStyle = shade(c, -0.1);
    ctx.fill();
    ctx.lineWidth = 1.2;
    ctx.strokeStyle = "#fbf6e8";
    ctx.stroke();
    ctx.restore();
    return;
  }
  // 远景小牌和城市名牌一样高（名牌：字号 10–14 + 上下边距 12），缩小地图时也看得清
  const fs = Math.round(Math.max(10, Math.min(14, 9 + z * 0.12)));
  const h = fs + 12;
  const w = Math.round(h * 1.4);
  ctx.shadowColor = "rgba(0,0,0,0.45)";
  ctx.shadowBlur = 3;
  ctx.shadowOffsetY = 1;
  ctx.beginPath();
  roundRect(ctx, x - w / 2, y - h / 2, w, h, 2.5);
  ctx.fillStyle = shade(c, -0.05);
  ctx.fill();
  ctx.shadowColor = "transparent";
  ctx.lineWidth = 2.4;
  ctx.strokeStyle = "rgba(16,13,9,0.85)";
  ctx.stroke();
  ctx.lineWidth = 1;
  ctx.strokeStyle = "rgba(255,248,230,0.9)";
  ctx.stroke();
  const ink = inkOn(c);
  if (military.length > 1) {
    ctx.font = `700 ${Math.round(h * 0.78)}px ${FONT}`;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillStyle = ink;
    ctx.fillText(String(military.length), x, y + 0.5);
  } else {
    drawSymbol(ctx, unitDef(top).icon, x, y, w * 0.62, h * 0.56, ink);
  }
  ctx.restore();
}

function drawArmyTokens(ctx, v) {
  // 从上到下画，下面一行的牌压住上面一行；城里的部队由城市标记代表
  const tiles = [...visibleEntries(G.stacks, v)].map(([t]) => t).sort((a, b) => a - b);
  for (const t of tiles) {
    if (G.cityAt.has(t)) continue;
    const st = G.stacks.get(t);
    const units = [...st.surface, ...st.air].filter((u) => !hiddenByMovement(u));
    if (!units.length) continue;
    const [wx, wy] = center(t);
    eachCopy(v, wx, wy, 16, (x, y) => drawToken(ctx, x, y, v.z, units));
  }
}

function drawUnits(ctx, v) {
  const z = v.z;
  if (z < (G.snap.stress ? 20 : LOD.counters)) {
    // 压力测试场景部队成千上万，远景仍只画小点
    if (!G.snap.stress) drawArmyTokens(ctx, v);
    else if (z >= LOD.armyDot) drawArmyDots(ctx, v);
    return;
  }
  const pad = z * 2;
  // 从上到下、从左到右画，下面一行的牌压住上面一行
  const tiles = [...visibleEntries(G.stacks,v)].map(([t])=>t).sort((a, b) => a - b);
  for (const t of tiles) {
    const original = G.stacks.get(t);
    const st = { surface: original.surface.filter(u=>!hiddenByMovement(u)), air: original.air.filter(u=>!hiddenByMovement(u)) };
    if (!st.surface.length && !st.air.length) continue;
    const [wx, wy] = center(t);
    const [ox, oy] = attackOffset(st.surface[0] ?? st.air[0], v);
    // 城里的部队画小一点、放在城市图形前面，不把城市整个挡住
    if (G.cityAt.has(t)) eachCopy(v, wx, wy, pad, (x, y) => drawStack(ctx, x + z * 0.68 + ox, y + z * 0.48 + oy, z * 0.6, st));
    else eachCopy(v, wx, wy, pad, (x, y) => drawStack(ctx, x + ox, y + z * 0.08 + oy, z, st));
  }
}

// ---------------------------------------------------------------- 选中与移动路线

function wrapDelta(dx, PW) {
  if (dx > PW / 2) return dx - PW;
  if (dx < -PW / 2) return dx + PW;
  return dx;
}

function drawRoute(ctx, v, u, dest, strong) {
  const [x0, y0] = center(u.tile);
  const tile = dest.row * G.W + dest.col;
  const [x1, y1] = center(tile);
  const dx = wrapDelta(x1 - x0, v.PW);
  const z = v.z;
  for (let k = v.k0; k <= v.k1; k++) {
    const ax = v.sx(x0, k), ay = v.sy(y0);
    const bx = v.sx(x0 + dx, k), by = v.sy(y1);
    if (Math.max(ax, bx) < -50 || Math.min(ax, bx) > v.vw + 50) continue;
    ctx.save();
    ctx.setLineDash([Math.max(4, z * 0.25), Math.max(3, z * 0.18)]);
    ctx.lineWidth = Math.max(1.5, Math.min(4, z * 0.1));
    ctx.strokeStyle = strong ? "rgba(255,212,121,0.95)" : "rgba(255,248,230,0.6)";
    ctx.beginPath();
    ctx.moveTo(ax, ay);
    ctx.lineTo(bx, by);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.beginPath();
    hexPath(ctx, bx, by, z * 0.85);
    ctx.stroke();
    const ang = Math.atan2(by - ay, bx - ax);
    const hl = Math.max(6, z * 0.3);
    ctx.beginPath();
    ctx.moveTo(bx - Math.cos(ang) * z * 0.85, by - Math.sin(ang) * z * 0.85);
    ctx.lineTo(bx - Math.cos(ang) * z * 0.85 - Math.cos(ang - 0.45) * hl, by - Math.sin(ang) * z * 0.85 - Math.sin(ang - 0.45) * hl);
    ctx.moveTo(bx - Math.cos(ang) * z * 0.85, by - Math.sin(ang) * z * 0.85);
    ctx.lineTo(bx - Math.cos(ang) * z * 0.85 - Math.cos(ang + 0.45) * hl, by - Math.sin(ang) * z * 0.85 - Math.sin(ang + 0.45) * hl);
    ctx.stroke();
    ctx.restore();
  }
}

function drawSelection(ctx, v) {
  if (!G.selected) return;
  const [wx, wy] = center(G.selected.tile);
  const z = v.z;
  eachCopy(v, wx, wy, z * 2, (x, y) => {
    ctx.save();
    ctx.beginPath();
    hexPath(ctx, x, y, z * 0.97);
    ctx.lineWidth = Math.max(1.5, Math.min(2.5, z * 0.035)) + 2;
    ctx.strokeStyle = "rgba(20,16,10,0.6)";
    ctx.stroke();
    ctx.lineWidth = Math.max(1.5, Math.min(2.5, z * 0.035));
    ctx.strokeStyle = "#ffd479";
    ctx.stroke();
    ctx.restore();
  });
}

function drawRoutes(ctx, v) {
  // 只画选中格子里的单位还没走完的路线
  const sel = G.selected;
  if (!sel) return;
  for (const u of G.stacks.get(sel.tile)?.surface ?? []) {
    const dest = destinationOf(u);
    if (dest) drawRoute(ctx, v, u, dest, true);
  }
}

// subtle = 地形模式：字小一些、画在部队下面，只是标出谁在哪，不盖住地图
function drawNationLabels(ctx, v, subtle = false) {
  ctx.save();
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.lineJoin = "round";
  // 大国先放；和已经放下的国名重叠就跳过（拉近以后有城市名牌接替）
  const placed = [];
  for (const L of [...G.labels].sort((a, b) => b.size - a.size)) {
    const fs = Math.round(Math.max(11, Math.min(subtle ? 17 : 30, L.size * v.z * (subtle ? 0.4 : 0.55))));
    ctx.font = `600 ${fs}px "Songti SC", "Noto Serif CJK SC", "Source Han Serif SC", ${FONT}`;
    const half = ctx.measureText(L.name).width / 2 + 4;
    eachCopy(v, L.x, L.y, 200, (x, y) => {
      const box = [x - half, y - fs * 0.65, x + half, y + fs * 0.65];
      if (placed.some((b) => box[0] < b[2] && box[2] > b[0] && box[1] < b[3] && box[3] > b[1])) return;
      placed.push(box);
      ctx.lineWidth = Math.max(3, fs * 0.22);
      ctx.strokeStyle = "rgba(251,246,232,0.9)";
      ctx.strokeText(L.name, x, y);
      ctx.fillStyle = shade(L.rgb, -0.55);
      ctx.fillText(L.name, x, y);
    });
  }
  ctx.restore();
}

export function drawOver(ctx, v) {
  if (!G.snap) return;
  if(G.layers.capacity)drawCapacity(ctx,v);
  drawRoads(ctx, v);
  if(G.layers.facilities && v.z>=12){
    for(const [tile,p] of visibleEntries(G.constructionAt,v)){
      const [wx,wy]=center(tile);
      eachCopy(v,wx,wy,v.z*1.2,(x,y)=>{
        ctx.save();hexPath(ctx,x,y,v.z*.9);ctx.fillStyle='rgba(193,158,83,.12)';ctx.fill();ctx.setLineDash([5,4]);ctx.strokeStyle='#dbc082';ctx.lineWidth=1.3;ctx.stroke();ctx.setLineDash([]);
        const z=v.z*.4;ctx.strokeStyle='#edcf8e';ctx.lineWidth=Math.max(1,z*.13);ctx.beginPath();ctx.moveTo(x-z,y+z*.7);ctx.lineTo(x-z,y-z);ctx.lineTo(x+z,y-z);ctx.lineTo(x+z,y+z*.7);ctx.moveTo(x-z,y);ctx.lineTo(x+z,y);ctx.moveTo(x-z,y+z*.7);ctx.lineTo(x+z,y-z);ctx.stroke();
        if(v.z>=24){ctx.font=`600 ${Math.max(10,Math.min(13,v.z*.3))}px ${FONT}`;ctx.textAlign='center';ctx.textBaseline='top';ctx.fillStyle='#f4e2b9';ctx.fillText(`施工 · ${p.turns_left} 回合`,x,y+z);}
        ctx.restore();
      });
    }
  }
  // 地形模式在城市名牌出现之前（最远景）也标国名，观战时一眼看出谁在哪；画在城市和部队下面
  if (!G.political && G.layers.territory && v.z < LOD.cityPlate) drawNationLabels(ctx, v, true);
  drawBattlePlans(ctx, v);
  if (G.layers.cities) drawCityMarkers(ctx, v);
  drawTactics(ctx, v);
  drawSelection(ctx, v);
  if (G.layers.units) {
    drawRoutes(ctx, v);
    drawUnits(ctx, v);
  }
  drawLevelBadges(ctx, v);
  if (G.layers.cities) drawCityPlates(ctx, v);
  drawEffects(ctx, v);
  if (G.political && v.z < POLITICAL_LABELS_BELOW && G.layers.territory) drawNationLabels(ctx, v);
}

// ---------------------------------------------------------------- 作战计划箭头（战役图风格）
// 每国的各条战线（server 端 controllers/rule_ai.py 的 _battle_plan，一条战线一场战役）：
// 战线（一排格子连成的粗线，锯齿朝向敌方）、主攻（粗箭头，战线中点 → 目标城）、
// 钳形两翼（战线两端 → 目标城左右后方，向外弯的箭头）；跨海作战另有航渡箭头（出发港 → 登陆场，长虚线）；
// 本回合野战部队的包抄 / 包围动作是细箭头。备战时淡、推进 / 航渡时中、围城时最实。
const PLAN_STYLE = {
  main: { w: 0.85, min: 8, head: 1.9 },
  flank: { w: 0.4, min: 4.5, head: 2.3, bend: 0.55, spread: 1.3 },
  assemble: { w: 0.2, min: 2.5, head: 3.2, dash: true },
  sail: { w: 0.45, min: 5, head: 2.2 },
  flank_move: { w: 0.15, min: 2, head: 3.4 },
  encircle: { w: 0.15, min: 2, head: 3.4 },
};
const PLAN_ALPHA = { 备战: 0.38, 整编: 0.45, 装船: 0.5, 推进: 0.6, 航渡: 0.6, 围城: 0.72 };

function drawBattlePlans(ctx, v) {
  if (!G.layers.plans || G.political) return;
  for (const n of G.nations.values()) {
    const st = n.strategy ?? {};
    // 战役图（苏式）：只画战线和实线的进攻箭头（主攻、跨海航渡），不画包围钳形、虚线和零碎的包抄小箭头
    for (const plan of st.battles ?? (st.battle ? [st.battle] : [])) {
      const alpha = PLAN_ALPHA[plan.phase] ?? 0.5;
      if (plan.line?.length > 1) drawFrontLine(ctx, v, plan, n.rgb, alpha);
      for (const a of plan.arrows ?? []) if (a.kind === 'main' || a.kind === 'sail') drawPlanArrow(ctx, v, a, n.rgb, alpha, false);
      const main = plan.arrows?.find(a => a.kind === 'main');
      if (main && v.z >= 5) drawPlanLabel(ctx, v, main, plan, n.rgb);
    }
  }
}

// 战线：沿一排格子中心连成的粗线，线上一串小三角（锯齿）朝向目标城，像战役图上的前线符号
function drawFrontLine(ctx, v, plan, rgb, alpha) {
  const pts = [];
  let [px] = center(plan.line[0]);
  for (const t of plan.line) {
    const [x, y] = center(t);
    px = px + wrapDelta(x - px, v.PW);
    pts.push([px, y]);
  }
  let [tx, ty] = center(plan.target_tile ?? plan.arrows?.find(a => a.kind === 'main')?.to ?? plan.line[0]);
  tx = pts[0][0] + wrapDelta(tx - pts[0][0], v.PW);
  const width = Math.max(3, v.z * 0.22), tooth = Math.max(4, v.z * 0.28);
  const color = shade(rgb, 0.12);
  for (let k = v.k0; k <= v.k1; k++) {
    const sp = pts.map(([x, y]) => [v.sx(x, k), v.sy(y)]);
    const xs = sp.map(p => p[0]), ys = sp.map(p => p[1]);
    if (Math.max(...xs) < -40 || Math.min(...xs) > v.vw + 40 || Math.max(...ys) < -40 || Math.min(...ys) > v.vh + 40) continue;
    const ox = v.sx(tx, k), oy = v.sy(ty);
    ctx.save();
    ctx.globalAlpha = alpha;
    ctx.lineJoin = 'round'; ctx.lineCap = 'round';
    ctx.beginPath(); sp.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
    ctx.strokeStyle = 'rgba(24,20,16,0.7)'; ctx.lineWidth = width + 3; ctx.stroke();
    ctx.strokeStyle = color; ctx.lineWidth = width; ctx.stroke();
    // 锯齿：每段中点一个小三角，朝目标城那一侧
    ctx.fillStyle = color;
    for (let i = 0; i + 1 < sp.length; i++) {
      const [ax, ay] = sp[i], [bx, by] = sp[i + 1];
      const mx = (ax + bx) / 2, my = (ay + by) / 2, len = Math.hypot(bx - ax, by - ay) || 1;
      let nx = -(by - ay) / len, ny = (bx - ax) / len;
      if (nx * (ox - mx) + ny * (oy - my) < 0) { nx = -nx; ny = -ny; }
      const ux = (bx - ax) / len, uy = (by - ay) / len;
      ctx.beginPath();
      ctx.moveTo(mx - ux * tooth * 0.6, my - uy * tooth * 0.6);
      ctx.lineTo(mx + nx * tooth, my + ny * tooth);
      ctx.lineTo(mx + ux * tooth * 0.6, my + uy * tooth * 0.6);
      ctx.closePath(); ctx.fill();
    }
    ctx.restore();
  }
}

function planGeometry(v, a) {
  let [ax, ay] = center(a.from);
  let [bx, by] = center(a.to);
  bx = ax + wrapDelta(bx - ax, v.PW);
  const len = Math.hypot(bx - ax, by - ay) || 1;
  const style = PLAN_STYLE[a.kind] ?? {}, side = a.bend ?? 0;
  // 两翼往各自那一侧错开出发，再向外弯，和主攻箭头分开成钳形
  const px = -(by - ay) / len, py = (bx - ax) / len;
  ax += px * side * (style.spread ?? 0); ay += py * side * (style.spread ?? 0);
  const bend = side * (style.bend ?? 0) * len;
  return { ax, ay, bx, by, len, mx: (ax + bx) / 2 + px * bend, my: (ay + by) / 2 + py * bend };
}

function drawPlanArrow(ctx, v, a, rgb, alpha, dashed) {
  const s = PLAN_STYLE[a.kind] ?? PLAN_STYLE.flank_move;
  const p = planGeometry(v, a);
  if (p.len < 0.5) return;
  const width = Math.max(s.min, v.z * s.w), head = width * s.head;
  const color = shade(rgb, 0.12);
  for (let k = v.k0; k <= v.k1; k++) {
    const x1 = v.sx(p.ax, k), y1 = v.sy(p.ay), x2 = v.sx(p.bx, k), y2 = v.sy(p.by), cx = v.sx(p.mx, k), cy = v.sy(p.my);
    const pad = head + 20;
    if (Math.max(x1, x2, cx) < -pad || Math.min(x1, x2, cx) > v.vw + pad || Math.max(y1, y2, cy) < -pad || Math.min(y1, y2, cy) > v.vh + pad) continue;
    // 线身止于箭头底部，箭头沿终点切线方向
    const tx = x2 - cx, ty = y2 - cy, tl = Math.hypot(tx, ty) || 1, ux = tx / tl, uy = ty / tl;
    const ex = x2 - ux * head * 0.85, ey = y2 - uy * head * 0.85;
    ctx.save();
    ctx.globalAlpha = alpha;
    ctx.lineCap = 'butt'; ctx.lineJoin = 'round';
    if (Array.isArray(s.dash)) ctx.setLineDash([width * s.dash[0], width * s.dash[1]]);
    else if (s.dash || dashed) ctx.setLineDash([width * 2.4, width * 1.4]);
    ctx.beginPath(); ctx.moveTo(x1, y1); ctx.quadraticCurveTo(cx, cy, ex, ey);
    ctx.strokeStyle = 'rgba(24,20,16,0.7)'; ctx.lineWidth = width + 3; ctx.stroke();
    ctx.strokeStyle = color; ctx.lineWidth = width; ctx.stroke();
    ctx.setLineDash([]);
    ctx.beginPath();
    ctx.moveTo(x2, y2);
    ctx.lineTo(ex - uy * head * 0.72, ey + ux * head * 0.72);
    ctx.lineTo(ex + uy * head * 0.72, ey - ux * head * 0.72);
    ctx.closePath();
    ctx.fillStyle = color; ctx.fill();
    ctx.strokeStyle = 'rgba(24,20,16,0.8)'; ctx.lineWidth = 1.5; ctx.stroke();
    ctx.restore();
  }
}

function drawPlanLabel(ctx, v, main, plan, rgb) {
  const p = planGeometry(v, main);
  const text = `${plan.name} · ${plan.phase}`;
  const fs = Math.round(Math.max(11, Math.min(16, v.z * 0.42)));
  // 名牌放在战线后方（主攻起点再往后一格多），不和前线的部队牌、城市名牌挤在一起
  const back = 2.2 / (p.len || 1);
  eachCopy(v, p.ax - (p.bx - p.ax) * back, p.ay - (p.by - p.ay) * back, 200, (x, y) => {
    ctx.save();
    ctx.font = `700 ${fs}px "Songti SC", "Noto Serif CJK SC", "Source Han Serif SC", ${FONT}`;
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.lineJoin = 'round';
    ctx.lineWidth = Math.max(3, fs * 0.3); ctx.strokeStyle = 'rgba(251,246,232,0.92)';
    ctx.strokeText(text, x, y);
    ctx.fillStyle = shade(rgb, -0.5); ctx.fillText(text, x, y);
    ctx.restore();
  });
}

function drawCapacity(ctx,v){
  const data=G.snap.settlement_regions;if(!data)return;
  const width=data.width,columns=Math.ceil(G.W/width);
  const zone=t=>Math.floor(Math.floor(t/G.W)/width)*columns+Math.floor((t%G.W)/width);
  const fills=new Map(),edges=new Path2D();
  for(const t of visibleTiles(v)){
    const region=zone(t),cap=data.capacities[region]??0,count=data.cities[region]??0;
    const color=!cap?'rgba(84,85,91,.2)':count>=cap?'rgba(194,99,73,.23)':`rgba(87,169,128,${.12+Math.min(8,cap)*.015})`;
    if(!fills.has(color))fills.set(color,new Path2D());
    const [wx,wy]=center(t);
    eachCopy(v,wx,wy,v.z*1.5,(x,y)=>{
      hexPath(fills.get(color),x,y,v.z);
      for(let d=0;d<6;d++)if(neighbor(t,d)<0||zone(neighbor(t,d))!==region){
        const [a,b]=EDGE_CORNERS[d];edges.moveTo(x+CORNERS[a][0]*v.z,y+CORNERS[a][1]*v.z);edges.lineTo(x+CORNERS[b][0]*v.z,y+CORNERS[b][1]*v.z);
      }
    });
  }
  ctx.save();for(const [color,path]of fills){ctx.fillStyle=color;ctx.fill(path);}
  ctx.strokeStyle='#d6dfc680';ctx.lineWidth=1;ctx.stroke(edges);
  ctx.font=`600 12px ${FONT}`;ctx.textAlign='center';ctx.textBaseline='middle';
  for(let r=0;r<G.H;r+=width)for(let c=0;c<G.W;c+=width){
    const region=Math.floor(r/width)*columns+Math.floor(c/width),t=Math.min(G.H-1,r+Math.floor(width/2))*G.W+Math.min(G.W-1,c+Math.floor(width/2));
    const [wx,wy]=center(t);eachCopy(v,wx,wy,70,(x,y)=>{const text=`城市 ${data.cities[region]??0} / ${data.capacities[region]??0}`;ctx.fillStyle='#192d2bee';ctx.fillRect(x-48,y-11,96,22);ctx.fillStyle='#e4e2cf';ctx.fillText(text,x,y);});
  }
  ctx.restore();
}

export const SYMBOLS = { drawSymbol };

// Tactical overlays use server-calculated tiles and paths, including formations and roads.
G.actor = null;
G.tactics = null;
G.actionMode = 'inspect';
G.hoverTile = null;
G.pending = null;
G.effects = [];
G.layers.roads = true;

export function setTactics(data) {
  G.tactics = data;
  G.pending = null;
}

function traceTiles(ctx, v, path, color, dashed = false) {
  if (!path?.length) return;
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = Math.max(2, Math.min(4, v.z * .075));
  ctx.lineCap = 'round';
  if (dashed) ctx.setLineDash([7, 5]);
  const [x0, y0] = center(path[0]);
  for (let k = v.k0; k <= v.k1; k++) {
    ctx.beginPath();
    let px = x0;
    path.forEach((tile, i) => {
      const [wx, wy] = center(tile);
      px = i ? px + wrapDelta(wx - px, v.PW) : wx;
      const x = v.sx(px, k), y = v.sy(wy);
      if (!i) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();
    const [wx, wy] = center(path.at(-1));
    const ex = v.sx(px, k), ey = v.sy(wy);
    ctx.setLineDash([]);
    ctx.beginPath(); ctx.arc(ex, ey, Math.max(4, v.z*.12), 0, Math.PI*2);
    ctx.fillStyle = color; ctx.fill();
  }
  ctx.restore();
}

function tacticalHex(ctx, v, tile, fill, stroke, width = 1) {
  const [wx, wy] = center(tile);
  eachCopy(v, wx, wy, v.z * 2, (x, y) => {
    ctx.beginPath(); hexPath(ctx, x, y, v.z * .96);
    if (fill) { ctx.fillStyle = fill; ctx.fill(); }
    if (stroke) { ctx.lineWidth = width; ctx.strokeStyle = stroke; ctx.stroke(); }
  });
}

function drawRangeBoundary(ctx, v, tiles, color) {
  const inside = new Set(tiles);
  const actor=G.units.get(G.actor)??G.cities.get(G.actor);
  if(actor)inside.add(actor.tile);
  const path = new Path2D(),r=v.z*.97;
  for(const tile of inside){
    const [wx,wy]=center(tile);
    eachCopy(v,wx,wy,v.z*2,(x,y)=>{
      for(let d=0;d<6;d++){
        if(inside.has(neighbor(tile,d)))continue;
        const [a,b]=EDGE_CORNERS[d];
        path.moveTo(x+CORNERS[a][0]*r,y+CORNERS[a][1]*r);
        path.lineTo(x+CORNERS[b][0]*r,y+CORNERS[b][1]*r);
      }
    });
  }
  ctx.save();ctx.lineJoin='round';ctx.lineCap='round';
  ctx.strokeStyle='rgba(25,43,45,.42)';ctx.lineWidth=4;ctx.stroke(path);
  ctx.strokeStyle=color;ctx.lineWidth=1.7;ctx.stroke(path);ctx.restore();
}

function drawTactics(ctx, v) {
  const t = G.tactics;
  if (!t || t.actor !== G.actor || v.z < 3) return;
  ctx.save();
  const attacking = G.actionMode === 'attack';
  const moving = G.actionMode === 'move' || G.actionMode === 'inspect';
  const targetSet = new Set(t.targets.map(x => x.tile));
  const blocked = new Set(t.obstructed);
  if (moving) {
    for (const tile of t.move_tiles ?? []) tacticalHex(ctx, v, tile, 'rgba(95,186,211,.17)', 'rgba(145,221,239,.32)', .65);
    drawRangeBoundary(ctx, v, t.move_tiles ?? [], '#a5e3ec');
  }
  if (attacking) {
    for (const tile of t.attack_tiles) {
      const obstructed = blocked.has(tile);
      tacticalHex(ctx, v, tile, obstructed ? 'rgba(132,139,142,.1)' : 'rgba(214,99,74,.13)',
        obstructed ? 'rgba(161,162,153,.3)' : 'rgba(235,135,102,.45)', .9);
    }
  }
  if (attacking) drawRangeBoundary(ctx, v, t.attack_tiles, '#e9b08a');
  for (const tile of targetSet) {
    tacticalHex(ctx, v, tile, 'rgba(213,79,55,.22)', '#ed9a78', Math.max(1.6, v.z*.035));
    const [wx, wy] = center(tile);
    eachCopy(v, wx, wy, v.z*2, (x,y) => {
      ctx.strokeStyle = '#ffe0b8'; ctx.lineWidth = 1.5;
      const r = Math.max(5, v.z*.14);
      ctx.beginPath(); ctx.arc(x,y-v.z*.56,r,0,Math.PI*2); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(x-r*1.5,y-v.z*.56);ctx.lineTo(x+r*1.5,y-v.z*.56);
      ctx.moveTo(x,y-v.z*.56-r*1.5);ctx.lineTo(x,y-v.z*.56+r*1.5);ctx.stroke();
    });
  }
  const tile = G.pending?.tile ?? G.hoverTile;
  if (tile != null) {
    if (moving && t.move_paths?.[tile]) traceTiles(ctx, v, t.move_paths[tile], '#a7e7f2');
    if (attacking && targetSet.has(tile)) {
      const actor = G.units.get(G.actor) ?? G.cities.get(G.actor);
      if (actor) traceTiles(ctx, v, [actor.tile, tile], '#ffbc87', true);
    }
  }
  ctx.restore();
}

function drawLevelBadges(ctx, v) {
  // 设施等级只标在选中和鼠标所在的格子上：每格都标的话满图时满屏角标，压住地块和兵牌
  if (v.z < 12) return;
  ctx.save(); ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
  const tiles = new Set([G.selected?.tile, G.hoverTile].filter((t) => t != null && G.facilityAt.has(t)));
  for (const tile of tiles) {
    const lv = facilityProgress(tile).level;
    const [wx, wy] = center(tile);
    eachCopy(v, wx, wy, v.z*2, (x,y) => {
      const s = Math.min(13, Math.max(8, v.z*.25));
      ctx.fillStyle = 'rgba(19,31,35,.9)';ctx.beginPath();
      roundRect(ctx,x-s,y+v.z*.56-s*.65,s*2,s*1.3,3);ctx.fill();
      ctx.font=`600 ${s*.85}px ${FONT}`;ctx.fillStyle='#e9d29b';ctx.fillText(`L${lv}`,x,y+v.z*.56);
    });
  }
  ctx.restore();
}

export function playEvents(events, before) {
  const now = performance.now();
  const old = new Map((before?.units ?? []).map(u => [u.id,u]));
  const combats = events.filter(e=>e.type==='combat');
  const destroyed = new Set(events.filter(e=>e.type==='unit_destroyed').map(e=>e.unit));
  const grouped = new Set();
  let combatIndex = 0;
  for (const e of events) {
    if (e.type === 'combat') {
      G.effects.push({type:'combat', ...e, start:now+combatIndex++*240, duration:1450,
        attackerOld:old.get(e.attacker), defenderOld:old.get(e.defender)});
    } else if (e.type === 'unit_destroyed') {
      const c = combats.find(c=>c.attacker===e.unit||c.defender===e.unit);
      G.effects.push({...e, type:'loss', unit:old.get(e.unit), start:now+(c?510:0), duration:1400});
    } else if (e.type === 'unit_moved' && e.path?.length > 1 && !destroyed.has(e.unit)) {
      const unit = old.get(e.unit);
      if (!unit) continue;
      const key = unit.formation ?? unit.id;
      if (grouped.has(key)) continue;
      grouped.add(key);
      const count = unit.formation == null ? 1 : [...old.values()].filter(u=>u.formation===unit.formation).length;
      const duration = Math.min(1500, 320 + e.path.length*140);
      G.effects.push({type:'movement', path:e.path, unit, count,
        start:now+(combats.some(c=>c.attacker===e.unit)?650:0), duration});
    } else if ((e.type === 'air_sortie' || e.type === 'air_returned') && e.source != null && e.source !== e.tile) {
      // 飞机起飞飞往目标上空 / 返航降落：飞机牌沿直线飞过去
      const unit = old.get(e.unit) ?? G.units.get(e.unit);
      if (unit) G.effects.push({type:'movement', path:[e.source, e.tile], unit, count:1, start:now, duration:900});
    } else if (e.type === 'facility_built') {
      G.effects.push({type:'label',tile:e.tile,text:`${G.rules.buildings[e.facility]?.name??'设施'}竣工`,start:now+150,duration:1500});
    } else if (e.type === 'pillaged') {
      G.effects.push({type:'label', tile:e.tile, text:`破坏 · ${e.target}`, start:now+200, duration:1600});
    } else if (e.type === 'city_independence') {
      G.effects.push({type:'label', tile:e.tile, text:`独立 · ${e.name}`, start:now+400, duration:2600});
    } else if (e.type === 'battle_plan' || e.type === 'battle_ended') {
      if (e.tile != null) G.effects.push({type:'label', tile:e.tile, text:e.type==='battle_plan'?`开战 · ${e.name}`:`${e.name} · ${e.result}`, start:now+300, duration:2200});
    } else if (e.type === 'level_up' || e.type === 'city_captured') {
      G.effects.push({type:'label', tile:e.tile, text:e.type==='level_up'?`晋升 · Lv.${e.level}`:'城市占领',
        start:now+550, duration:1600});
    }
  }
  G.effects = G.effects.slice(-100);
}

const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)');
export function tickEffects(now) {
  const had = G.effects.length > 0;
  G.effects = G.effects.filter(e=>now < e.start+e.duration);
  return had;
}

function movementPosition(e, progress) {
  const step = Math.min(e.path.length-1.001, Math.max(0, progress)*(e.path.length-1));
  const i = Math.floor(step), f=step-i;
  const [ax,ay]=center(e.path[i]),[bx,by]=center(e.path[i+1]);
  return [ax+wrapDelta(bx-ax,SQRT3*G.W)*f,ay+(by-ay)*f];
}

function hiddenByMovement(unit) {
  if (reducedMotion.matches) return false;
  return G.effects.some(e=>e.type==='movement' &&
    (e.unit.id===unit.id || (unit.formation!=null && e.unit.formation===unit.formation)));
}

function attackOffset(unit, v) {
  if (reducedMotion.matches) return [0,0];
  const now = performance.now();
  const shot = G.effects.find(e => e.type === 'combat' && e.attacker === unit.id);
  if (shot) {
    const t = (now-shot.start)/shot.duration;
    if (t>=0 && t<.5 && shot.kind!=='air') {
      const [ax,ay]=center(shot.source),[bx,by]=center(shot.tile);
      const f = shot.kind==='melee' ? Math.sin(t/.5*Math.PI)*.28 : -Math.sin(Math.min(1,t/.25)*Math.PI)*.07;
      return [wrapDelta(bx-ax,v.PW)*v.z*f,(by-ay)*v.z*f];
    }
  }
  const hit = G.effects.find(e => e.type === 'combat' && e.defender === unit.id);
  if (hit) {
    const t=(now-hit.start)/hit.duration;
    if(t>.33&&t<.58)return[Math.sin((t-.33)*95)*3*(1-(t-.33)/.25),0];
  }
  return [0,0];
}

function drawEffects(ctx,v) {
  const now=performance.now();
  ctx.save();
  for(const e of G.effects){
    const t=(now-e.start)/e.duration;
    if(t<0 && e.type==='loss' && e.unit) {
      const [wx,wy]=center(e.tile);
      eachCopy(v,wx,wy,v.z*2,(x,y)=>drawStack(ctx,x,y+v.z*.08,v.z,
        e.unit.layer==='air'?{surface:[],air:[e.unit]}:{surface:[e.unit],air:[]}));
    }
    if(t<0||t>1)continue;
    if(e.type==='movement'){
      if(reducedMotion.matches)continue;
      const [wx,wy]=movementPosition(e,t*t*(3-2*t));
      const nation=G.nations.get(e.unit.nation);
      if(!nation)continue;
      // 远景、中景用和静止部队一样的小兵牌，免得移动时突然冒出一张大牌
      const far=!G.snap.stress&&v.z<LOD.counters;
      eachCopy(v,wx,wy,Math.max(16,v.z*2),(x,y)=>far?drawToken(ctx,x,y,v.z,Array(e.count).fill(e.unit)):drawStack(ctx,x,y+v.z*.08,v.z,
        e.unit.layer==='air'?{surface:[],air:[e.unit]}:{surface:Array(e.count).fill(e.unit),air:[]}));
      continue;
    }
    const [wx,wy]=center(e.tile);
    eachCopy(v,wx,wy,v.z*3+150,(x,y)=>{
      if(e.type==='combat'){
        const [sx,sy]=center(e.source);
        const ax=x-wrapDelta(wx-sx,v.PW)*v.z,ay=y-(wy-sy)*v.z;
        const flight=Math.min(1,t/.36);
        if(!reducedMotion.matches&&t<.36&&e.kind!=='melee'){
          const px=ax+(x-ax)*flight,py=ay+(y-ay)*flight-Math.sin(flight*Math.PI)*v.z*(e.kind==='air'?1.4:.6);
          ctx.strokeStyle='rgba(255,222,159,.75)';ctx.lineWidth=2;
          ctx.beginPath();ctx.moveTo(px-(x-ax)*.06,py-(y-ay)*.06);ctx.lineTo(px,py);ctx.stroke();
          if(e.kind==='air'){
            // 空袭：飞机牌从基地起飞、飞到目标上空（不是一个小箭头）
            if(e.attackerOld)drawStack(ctx,px,py,v.z*.85,{surface:[],air:[e.attackerOld]});
          }else{ctx.fillStyle='#ffdda0';ctx.beginPath();ctx.arc(px,py,3,0,Math.PI*2);ctx.fill();}
        }
        const impact=(t-.33)/.67;
        if(impact<0)return;
        const r=v.z*(.15+impact*.7);
        ctx.globalAlpha=1-impact;
        ctx.strokeStyle='#edb37f';ctx.lineWidth=Math.max(1,3*(1-impact));
        ctx.beginPath();ctx.arc(x,y,r,0,Math.PI*2);ctx.stroke();
        if(!reducedMotion.matches){
          for(let j=0;j<12;j++){
            const a=j*2.399+e.tile*.1,dist=(.1+impact)*v.z*(.3+(j%4)*.14);
            ctx.fillStyle=j%3?'#e8b47e':'#8c7661';ctx.beginPath();ctx.arc(x+Math.cos(a)*dist,y+Math.sin(a)*dist-impact*v.z*.3,Math.max(1,3*(1-impact)),0,Math.PI*2);ctx.fill();
          }
        }
        ctx.globalAlpha=1-Math.max(0,(impact-.55)/.45);
        effectText(ctx,`−${e.damage}`,x,y-v.z*.4-impact*30,'#ffd1aa',18);
        if(e.retaliation)effectText(ctx,`−${e.retaliation}`,ax,ay-v.z*.4-impact*25,'#f3c394',15);
        // 飞机返航降落：打完沿原路飞回基地（被击落的不画）
        if(e.kind==='air'&&e.attacker_hp>0&&!reducedMotion.matches&&t>.6){
          const back=Math.min(1,(t-.6)/.36);
          const px=x+(ax-x)*back,py=y+(ay-y)*back-Math.sin(back*Math.PI)*v.z*1.4;
          ctx.globalAlpha=1;
          if(e.attackerOld)drawStack(ctx,px,py,v.z*.85,{surface:[],air:[e.attackerOld]});
        }
      }else if(e.type==='loss'){
        ctx.globalAlpha=1-t;
        if(e.unit){const n=G.nations.get(e.unit.nation);if(n){
          ctx.save();ctx.translate(x,y-t*22);ctx.rotate(reducedMotion.matches?0:t*.18);
          drawStack(ctx,0,0,v.z*.7,{surface:e.unit.layer==='surface'?[e.unit]:[],air:e.unit.layer==='air'?[e.unit]:[]});ctx.restore();
        }}
        if (!reducedMotion.matches) { for(let j=0;j<5;j++){ctx.fillStyle='rgba(91,92,78,.27)';ctx.beginPath();ctx.arc(x+Math.sin(j*3.1)*v.z*t*.3,y-(t*.65+j*.04)*v.z,v.z*(.12+t*.22),0,Math.PI*2);ctx.fill();}}
        ctx.strokeStyle='#e4ab88';ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(x-8,y-8);ctx.lineTo(x+8,y+8);ctx.moveTo(x+8,y-8);ctx.lineTo(x-8,y+8);ctx.stroke();
        effectText(ctx,'部队损失',x,y+v.z*.45-t*16,'#e8bca0',13);
      }else if(e.type==='label'){
        ctx.globalAlpha=Math.min(1,t*5, (1-t)*4);effectText(ctx,e.text,x,y-v.z*.75-t*18,'#f5dd98',14);
      }
      ctx.globalAlpha=1;
    });
  }
  ctx.restore();
}

function effectText(ctx,text,x,y,color,size){
  ctx.font=`700 ${size}px ${FONT}`;ctx.textAlign='center';ctx.textBaseline='middle';
  ctx.lineWidth=4;ctx.strokeStyle='rgba(21,24,26,.9)';ctx.strokeText(text,x,y);
  ctx.fillStyle=color;ctx.fillText(text,x,y);
}

// Roads belong to their own layer and are painted above rivers and developed hexes.
G.roadPlan = null;
G.roadStart = null;
G.roadNation = null;

function roadSegment(ctx,v,seg,preview=false){
  const [ax,ay]=center(seg.a),[bx0,by]=center(seg.b);
  const bx=ax+wrapDelta(bx0-ax,v.PW);
  const z=v.z, width=Math.max(1.6,Math.min(9,z*.13));
  for(let k=v.k0;k<=v.k1;k++){
    const x0=v.sx(ax,k),y0=v.sy(ay),x1=v.sx(bx,k),y1=v.sy(by);
    if(Math.max(x0,x1)<-z||Math.min(x0,x1)>v.vw+z||Math.max(y0,y1)<-z||Math.min(y0,y1)>v.vh+z)continue;
    ctx.save();ctx.lineCap='round';ctx.lineJoin='round';
    if(preview){
      ctx.strokeStyle=seg.kind==='bridge'?'#b0dfe1':seg.kind==='tunnel'?'#cfb2df':'#eed39b';
      ctx.setLineDash([6,4]);ctx.lineWidth=Math.max(2,width*.75);
      ctx.beginPath();ctx.moveTo(x0,y0);ctx.lineTo(x1,y1);ctx.stroke();ctx.restore();continue;
    }
    if(seg.kind==='tunnel'){
      const infoA=v.tile?.(seg.a%G.W,Math.floor(seg.a/G.W));
      const infoB=v.tile?.(seg.b%G.W,Math.floor(seg.b/G.W));
      const aMountain=infoA?.role==='mountains',bMountain=infoB?.role==='mountains';
      const dx=x1-x0,dy=y1-y0;
      const start=aMountain?0:.5,end=bMountain?1:.5;
      const stroke=(a,b,color,lw,dash=[])=>{
        ctx.setLineDash(dash);ctx.strokeStyle=color;ctx.lineWidth=lw;
        ctx.beginPath();ctx.moveTo(x0+dx*a,y0+dy*a);ctx.lineTo(x0+dx*b,y0+dy*b);ctx.stroke();
      };
      if(start>0){stroke(0,start,'rgba(37,36,31,.7)',width+2);stroke(0,start,'#bbaa87',width);}
      if(end<1){stroke(end,1,'rgba(37,36,31,.7)',width+2);stroke(end,1,'#bbaa87',width);}
      stroke(start,end,'rgba(37,42,40,.8)',Math.max(1.5,width*.65),[4,4]);ctx.setLineDash([]);
      if(z>=14){
        for(const f of [!aMountain?start:null,!bMountain?end:null]){
          if(f==null)continue;
          ctx.save();ctx.translate(x0+dx*f,y0+dy*f);ctx.rotate(Math.atan2(dy,dx));
          ctx.shadowColor='rgba(18,26,23,.4)';ctx.shadowBlur=2;ctx.shadowOffsetY=2;
          ctx.fillStyle='#969681';ctx.strokeStyle='#454e46';ctx.lineWidth=1;
          ctx.beginPath();roundRect(ctx,-4,-width*1.15,8,width*2.3,2);ctx.fill();ctx.stroke();
          ctx.shadowBlur=0;ctx.shadowOffsetY=0;
          ctx.fillStyle='#26312b';ctx.fillRect(-3,-width*.62,6,width*1.24);
          ctx.fillStyle='#b9b69e';ctx.fillRect(-4,-width*1.15,8,2);ctx.restore();
        }
      }
      ctx.restore();continue;
    }
    // Low-key compact gravel surface; the darker edge preserves contrast on farms.
    ctx.strokeStyle='rgba(37,36,31,.7)';ctx.lineWidth=width+2;
    ctx.beginPath();ctx.moveTo(x0,y0);ctx.lineTo(x1,y1);ctx.stroke();
    ctx.strokeStyle='#bbaa87';ctx.lineWidth=width;
    ctx.stroke();
    if(z>=16){ctx.strokeStyle='rgba(237,222,185,.45)';ctx.lineWidth=Math.max(.65,width*.2);ctx.stroke();}
    if(seg.kind==='bridge'||seg.kind==='causeway'){
      const dx=x1-x0,dy=y1-y0,len=Math.hypot(dx,dy),nx=-dy/len,ny=dx/len;
      const start=seg.water?0:.33,end=seg.water?1:.67;
      ctx.lineCap='butt';ctx.shadowColor='rgba(20,42,44,.45)';ctx.shadowBlur=3;ctx.shadowOffsetY=3;
      ctx.strokeStyle='#443d31';ctx.lineWidth=width+5;
      ctx.beginPath();ctx.moveTo(x0+dx*start,y0+dy*start);ctx.lineTo(x0+dx*end,y0+dy*end);ctx.stroke();
      ctx.shadowBlur=0;ctx.shadowOffsetY=0;
      ctx.strokeStyle=seg.kind==='bridge'?'#cbbb9e':'#a9a99a';ctx.lineWidth=width+2;ctx.stroke();
      if(z>=12){
        // Railings and cross-planks distinguish a bridge from an ordinary road.
        ctx.strokeStyle='#62594a';ctx.lineWidth=1.2;
        for(const side of [-1,1]){
          ctx.beginPath();ctx.moveTo(x0+dx*start+nx*width*.65*side,y0+dy*start+ny*width*.65*side);
          ctx.lineTo(x0+dx*end+nx*width*.65*side,y0+dy*end+ny*width*.65*side);ctx.stroke();
        }
        const steps=Math.max(2,Math.round(len*.34/5));
        for(let i=0;i<=steps;i++){
          const f=start+(end-start)*i/steps,px=x0+dx*f,py=y0+dy*f;
          ctx.beginPath();ctx.moveTo(px-nx*width*.55,py-ny*width*.55);ctx.lineTo(px+nx*width*.55,py+ny*width*.55);ctx.stroke();
        }
      }
    }
    ctx.restore();
  }
}

// 道路按缩放分级：远景不画（几万段道路在远处会糊成一层斜线），中景细淡线，近景完整路面和桥梁
const ROAD_FROM = 5;
function drawRoads(ctx,v){
  if(!G.layers.roads||v.z<ROAD_FROM)return;
  ctx.save();
  const roads=new Set();
  for(const [,segments] of visibleEntries(G.roadsAt,v))for(const seg of segments)roads.add(seg);
  if(v.z<10) {
    const paths=new Map();
    for(const seg of roads) {
      if(!paths.has(seg.kind))paths.set(seg.kind,new Path2D());
      const path=paths.get(seg.kind),[ax,ay]=center(seg.a),[bx,by]=center(seg.b);
      const dx=wrapDelta(bx-ax,v.PW);
      for(let k=v.k0;k<=v.k1;k++) {
        const x0=v.sx(ax,k),y0=v.sy(ay),x1=v.sx(ax+dx,k),y1=v.sy(by);
        if(Math.max(x0,x1)<-v.z||Math.min(x0,x1)>v.vw+v.z||Math.max(y0,y1)<-v.z||Math.min(y0,y1)>v.vh+v.z)continue;
        path.moveTo(x0,y0);path.lineTo(x1,y1);
      }
    }
    ctx.lineCap='round';
    ctx.globalAlpha=Math.min(1,0.35+(v.z-ROAD_FROM)*0.13);
    for(const [kind,path] of paths) {
      ctx.setLineDash(kind==='tunnel'?[3,3]:[]);
      ctx.lineWidth=kind==='bridge'?2:1;
      ctx.strokeStyle=kind==='bridge'?'#dccca8':kind==='tunnel'?'#776b79':kind==='causeway'?'#b3ae95':'#ac9b7b';
      ctx.stroke(path);
    }
    ctx.setLineDash([]);
  }else for(const seg of roads)roadSegment(ctx,v,seg);
  if(G.roadPlan?.segments)for(const seg of G.roadPlan.segments)roadSegment(ctx,v,seg,true);
  if(G.actionMode==='road'&&G.roadStart!=null)tacticalHex(ctx,v,G.roadStart,'rgba(220,197,143,.18)','#e3c58a',2);
  ctx.restore();
}
