// Project Sekai 地图浏览器
// - 数据按 32×32 分块从服务器加载，只加载看得见的块
// - 远景：每块预先画成位图（按缩放层级缓存）；近景：逐格矢量绘制
// - 画风：山体阴影 + 山脉/丘陵/森林小图形 + 平滑河流，默认不画格子线，信息按缩放逐级出现
// - 对局图层（领土、城市、单位牌）由外部通过 setOverlay 挂进来，在地形之上、颗粒暗角之下绘制

const SQRT3 = Math.sqrt(3);
const CORNERS = [0, 1, 2, 3, 4, 5].map((i) => {
  const a = ((60 * i - 30) * Math.PI) / 180;
  return [Math.cos(a), Math.sin(a)];
});
const EDGE_CORNERS = [[0, 1], [5, 0], [4, 5], [3, 4], [2, 3], [1, 2]];
const DIRS = [[1, 0], [1, -1], [0, -1], [-1, 0], [-1, 1], [0, 1]]; // 轴向坐标方向：E NE NW W SW SE

const TIERS = [3, 6, 12, 20]; // 位图缓存的格子半径（物理像素）
const VECTOR_FROM = 16; // 物理像素半径超过它就逐格矢量绘制（清晰的格子 + 细节贴图）
const DETAIL = { coast: 4.5, borders: 2.5, glyphs: 18, resources: 9, grid: 14 };
const RING = 2; // 位图多画一圈邻块的格子，柔化后裁掉，避免块与块之间出现接缝
const MAX_BITMAP_PIXELS = 32e6; // 位图缓存上限（约 128 MB）；远景位图很小，近景位图很大
const MAX_FETCH = 24; // 同时在途的分块请求：精细地球有 375 块，请求处理要等帧间空隙，并发少了加载很慢
const CHUNK_BYTES_PER_TILE = 19; // 与 server/mapdata.py 的分块格式一致
const DATA_VERSION = new URL(import.meta.url).searchParams.get("d") ?? "";
const BAKE_MS = 12; // 每帧烤位图的时间预算（至少烤一块）
const LOAD_REDRAW_MS = 120; // 加载中（分块到达 / 位图烤好）整屏重画的最短间隔；交互引起的重画不受限

const DEEP_WATER = [34, 66, 88];
const BACKGROUND = "#0f1720";
// "国家"模式的底图：不画地形，陆地、海洋各一种平涂颜色（国家领土由对局图层铺色）
const PLAIN_LAND = "#d8d1bf";
const PLAIN_SEA = "#3d5a6c";

// ---------------------------------------------------------------- 状态

const S = {
  canvas: null,
  ctx: null,
  dpr: 1,
  vw: 0,
  vh: 0,
  query: "",
  gen: 0,
  meta: null,
  W: 0,
  H: 0,
  CS: 32,
  PW: 1,
  chunks: new Map(),
  pending: new Set(),
  inflight: 0,
  queue: [],
  bakeQueue: [], // 待烤的位图 {ch, tier}，每次整屏重画时按可见范围重新排
  bakeKeys: new Set(),
  progress: false, // 加载有进展（新分块到达 / 位图烤好），需要重画但可以限频
  lastRender: 0,
  renderAt: 0,
  renderCost: 0, // 上一次整屏重画的实际耗时（含浏览器光栅化），用来决定加载中多久重画一次
  edgesStale: false,
  bitmaps: new Map(),
  bitmapPixels: 0,
  scaledBitmaps: new Map(),
  scaledPixels: 0,
  frameNo: 0,
  cam: { x: 0, y: 0, z: 4 },
  layers: { areas: true, rivers: true, resources: false, starts: true, grid: false, wind: false, currents: false },
  mode: "terrain",
  lut: null,
  roles: {},
  rivers: [],
  hover: null,
  areaEdges: null,
  dirty: true,
  onHover: null,
  onClick: null,
  onStatus: null,
  overlay: null, // { under(ctx, view), over(ctx, view) }：under 画在河流之下，over 画在最上层
};

// ---------------------------------------------------------------- 颜色

function hexRgb(c) {
  return [1, 3, 5].map((k) => parseInt(c.slice(k, k + 2), 16));
}
function rgb(c) {
  return `rgb(${c[0] | 0},${c[1] | 0},${c[2] | 0})`;
}
function mix(a, b, t) {
  return [0, 1, 2].map((k) => a[k] + (b[k] - a[k]) * t);
}

function buildLut(meta) {
  const terrains = meta.ruleset.terrain;
  const biomes = meta.ruleset.biomes;
  const index = Object.fromEntries(terrains.map((t, i) => [t.id, i]));
  S.roles = Object.fromEntries(Object.entries(biomes).map(([role, id]) => [role, index[id]]));
  S.roleOf = [];
  for (const [role, ti] of Object.entries(S.roles)) if (ti !== undefined) S.roleOf[ti] = role;
  const shallow = hexRgb(terrains[S.roles.coast].color);
  S.lut = terrains.map((t, ti) => {
    const base = hexRgb(t.color);
    const out = [];
    for (let q = 0; q < 64; q++) {
      const tone = (q + 0.5) / 64;
      let c;
      if (ti === S.roles.coast || ti === S.roles.ocean) {
        c = mix(shallow, DEEP_WATER, Math.pow(Math.min(1, tone * 1.4), 0.8));
      } else if (t.water) {
        c = base;
      } else {
        // 山脉、丘陵的明暗对比更强，远看能看出山体的立体感；平地只有轻微起伏
        const strength = ti === S.roles.mountains || ti === S.roles.volcano ? 0.9 : ti === S.roles.hills ? 0.6 : 0.4;
        const f = 1 + strength * (tone - 0.5);
        c = base.map((v) => Math.min(255, v * f));
      }
      out.push(rgb(c));
    }
    return out;
  });
}

// ---------------------------------------------------------------- 图层（地图模式）

// 数据图层的色带：[位置 0..1, 颜色]；ticks 用于图例
export const MODES = {
  terrain: { name: "地形" },
  nations: { name: "国家", plain: true }, // 只显示国家：底图平涂，不画地形
  temperature: {
    name: "气温",
    stops: [[0, "#2b3f7a"], [0.18, "#4f7fc4"], [0.36, "#a9cde3"], [0.5, "#eef0dc"], [0.62, "#f3d58e"], [0.76, "#ec9a55"], [0.9, "#c4473a"], [1, "#7a1e2c"]],
    ticks: ["-40°C", "-20°C", "0°C", "20°C", "40°C"],
  },
  rainfall: {
    name: "降雨",
    stops: [[0, "#b98a52"], [0.2, "#dcc286"], [0.4, "#c9d48d"], [0.6, "#86b878"], [0.8, "#3f8f72"], [1, "#1c5f73"]],
    ticks: ["0", "225", "900", "2000", "3600 mm"],
  },
  elevation: {
    name: "海拔",
    stops: [[0, "#5f8f55"], [0.1, "#93ad66"], [0.25, "#d2c888"], [0.45, "#c19a62"], [0.65, "#8f6c52"], [0.85, "#cfc8bf"], [1, "#ffffff"]],
    water: [[0, "#a7cfdf"], [0.35, "#5f97b8"], [1, "#183754"]],
    ticks: ["0 m", "1500 m", "3000 m", "4500 m", "6000 m"],
  },
  wind: {
    name: "风",
    stops: [[0, "#3d4a63"], [0.2, "#5b7aa3"], [0.4, "#8fb7c9"], [0.6, "#d9e2c8"], [0.8, "#efc27a"], [1, "#d9764a"]],
    ticks: ["无风", "", "中", "", "强"],
  },
  areas: { name: "大区" },
  plates: {
    name: "板块",
    swatches: [["#c9a77c", "大陆板块"], ["#4f6f86", "大洋板块"], ["#d0573f", "汇聚（造山 / 海沟）"], ["#57b8c4", "张裂（裂谷 / 洋中脊）"], ["#d8b54e", "平移"]],
  },
};
const BOUNDARY_COLORS = [null, "#d0573f", "#57b8c4", "#d8b54e"];
const SHADES = 16;

function ramp(stops) {
  const cols = stops.map(([t, c]) => [t, hexRgb(c)]);
  const out = [];
  for (let v = 0; v < 256; v++) {
    const t = v / 255;
    let k = 0;
    while (k < cols.length - 2 && t > cols[k + 1][0]) k++;
    const [t0, c0] = cols[k];
    const [t1, c1] = cols[k + 1];
    out.push(mix(c0, c1, Math.max(0, Math.min(1, (t - t0) / (t1 - t0 || 1)))));
  }
  return out;
}

// lut[v][shade] = CSS 颜色；shade 来自山体阴影，让数据图层也保留一点立体感
function shadedLut(rgbs, strength) {
  return rgbs.map((c) => {
    const row = [];
    for (let k = 0; k < SHADES; k++) {
      const f = 1 + strength * ((k + 0.5) / SHADES - 0.5);
      row.push(rgb(c.map((v) => Math.min(255, v * f))));
    }
    return row;
  });
}

const modeLuts = {};
function modeLut(mode) {
  if (!modeLuts[mode]) {
    const m = MODES[mode];
    modeLuts[mode] = {
      land: shadedLut(ramp(m.stops), mode === "elevation" ? 0.55 : 0.25),
      water: shadedLut(ramp(m.water ?? [[0, "#4a6576"], [1, "#22394a"]]), 0),
    };
  }
  return modeLuts[mode];
}

const areaColors = new Map();
function areaColor(area) {
  let c = areaColors.get(area);
  if (!c) {
    const h = hash(area + 7919);
    if (area >= S.meta.land_areas) {
      c = `hsl(205, 26%, ${26 + (h % 7)}%)`;
    } else {
      // 金角分布色相，相邻编号的大区颜色差得远
      const hue = (area * 137.508) % 360;
      c = `hsl(${hue.toFixed(1)}, ${20 + (h % 10)}%, ${60 + ((h >>> 8) % 9)}%)`;
    }
    areaColors.set(area, c);
  }
  return c;
}

function isWater(ti) {
  return S.meta.ruleset.terrain[ti].water;
}

function fillColor(t) {
  const ch = t.ch, li = t.li;
  const ti = ch.terrain[li];
  const mode = S.mode;
  if (mode === "terrain") return S.lut[ti][ch.tone[li] >> 2];
  if (mode === "nations") return isWater(ti) ? PLAIN_SEA : PLAIN_LAND;
  if (mode === "areas") return areaColor(ch.area[li]);
  if (mode === "plates") return BOUNDARY_COLORS[ch.boundary[li]] ?? plateColor(ch.plate[li]);
  const water = isWater(ti);
  const shade = water ? 0 : Math.min(SHADES - 1, (ch.tone[li] * SHADES) >> 8);
  const lut = modeLut(mode);
  if (mode === "wind") {
    // 风速：海面和陆地用同一条色带，才能看出陆地摩擦、山后尾流
    const v = Math.min(255, Math.round((ch.wspd[li] / (1.1 * S.meta.wind_speed_scale)) * 255));
    return lut.land[v][shade];
  }
  if (water && mode === "temperature") {
    // 海面温度：与陆地同一条色带、稍暗，暖流 / 寒流一眼可见
    const v = Math.max(0, Math.min(255, Math.round(((ch.temp[li] + 40) / 80) * 255)));
    return lut.land[v][3];
  }
  if (water) {
    const v = mode === "elevation" ? Math.min(255, Math.round((-ch.height[li] / 5500) * 255)) : Math.min(255, ch.tone[li]);
    return lut.water[v][0];
  }
  let v;
  if (mode === "temperature") v = Math.round(((ch.temp[li] + 40) / 80) * 255);
  else if (mode === "rainfall") v = Math.round(Math.sqrt(ch.rain[li] / 3600) * 255);
  else v = Math.round((ch.height[li] / 6000) * 255);
  return lut.land[Math.max(0, Math.min(255, v))][shade];
}

const plateColors = new Map();
function plateColor(p) {
  let c = plateColors.get(p);
  if (!c) {
    const info = S.meta.plates[p];
    const h = hash(p + 101);
    c = info?.continental
      ? `hsl(${28 + (h % 26)}, ${24 + ((h >>> 5) % 12)}%, ${56 + ((h >>> 9) % 12)}%)`
      : `hsl(${198 + (h % 18)}, ${22 + ((h >>> 5) % 10)}%, ${30 + ((h >>> 9) % 10)}%)`;
    plateColors.set(p, c);
  }
  return c;
}

// 地面风：分块里每格的去向（wdir / 256 圈，从正东起顺时针）和风速
function windOf(td) {
  const a = (td.ch.wdir[td.li] / 256) * 2 * Math.PI;
  const speed = td.ch.wspd[td.li] / S.meta.wind_speed_scale;
  return [Math.cos(a) * speed, Math.sin(a) * speed, speed];
}

const COMPASS = ["东", "东北", "北", "西北", "西", "西南", "南", "东南"];
// 风按"从哪里吹来"命名：向东吹的是西风
export function windName(dx, dy) {
  const from = Math.atan2(dy, -dx); // 来向的角度（从正东起逆时针；屏幕 y 向下，所以北分量 = dy）
  const k = ((Math.round(from / (Math.PI / 4)) % 8) + 8) % 8;
  return `${COMPASS[k]}风`;
}

function hash(i) {
  let h = Math.imul(i ^ 0x9e3779b9, 0x85ebca6b);
  h ^= h >>> 13;
  h = Math.imul(h, 0xc2b2ae35);
  return (h ^ (h >>> 16)) >>> 0;
}

// ---------------------------------------------------------------- 数据

function chunkKey(cx, cy) {
  return cx + "," + cy;
}

function tileData(col, row) {
  if (row < 0 || row >= S.H) return null;
  col = ((col % S.W) + S.W) % S.W;
  const ch = S.chunks.get(chunkKey((col / S.CS) | 0, (row / S.CS) | 0));
  if (!ch) return null;
  return { ch, li: (row - ch.r0) * ch.cw + (col - ch.c0), col, row };
}

function requestChunk(cx, cy) {
  const key = chunkKey(cx, cy);
  if (S.chunks.has(key) || S.pending.has(key)) return;
  S.pending.add(key);
  S.queue.push([cx, cy]);
  pump();
}

function pump() {
  while (S.inflight < MAX_FETCH && S.queue.length) {
    const [cx, cy] = S.queue.shift();
    const gen = S.gen;
    S.inflight++;
    fetch(`/api/world/chunk?${S.query}&cx=${cx}&cy=${cy}`)
      .then((r) => {
        if (!r.ok) throw new Error(r.status);
        return r.arrayBuffer();
      })
      .then((buf) => {
        if (gen !== S.gen) return;
        const c0 = cx * S.CS;
        const r0 = cy * S.CS;
        const cw = Math.min(S.CS, S.W - c0);
        const chh = Math.min(S.CS, S.H - r0);
        const n = cw * chh;
        if (buf.byteLength !== CHUNK_BYTES_PER_TILE * n) {
          throw new Error(`分块 ${cx},${cy} 长度 ${buf.byteLength}，应为 ${CHUNK_BYTES_PER_TILE * n}`);
        }
        const u8 = new Uint8Array(buf);
        S.chunks.set(chunkKey(cx, cy), {
          cx, cy, c0, r0, cw, ch: chh,
          area: new Uint16Array(buf.slice(0, 2 * n)),
          height: new Int16Array(buf.slice(2 * n, 4 * n)),
          rain: new Uint16Array(buf.slice(4 * n, 6 * n)),
          terrain: u8.subarray(6 * n, 7 * n),
          resource: u8.subarray(7 * n, 8 * n),
          tone: u8.subarray(8 * n, 9 * n),
          aborder: u8.subarray(9 * n, 10 * n),
          coast: u8.subarray(10 * n, 11 * n),
          temp: new Int8Array(buf.slice(11 * n, 12 * n)),
          plate: u8.subarray(12 * n, 13 * n),
          boundary: u8.subarray(13 * n, 14 * n),
          wdir: u8.subarray(14 * n, 15 * n),
          wspd: u8.subarray(15 * n, 16 * n),
          cdir: u8.subarray(16 * n, 17 * n),
          cspd: u8.subarray(17 * n, 18 * n),
          canom: new Int8Array(buf.slice(18 * n, 19 * n)),
        });
        invalidateNeighbors(cx, cy);
        if (S.hover) S.edgesStale = true; // 悬停大区的轮廓到重画时再算，不必每到一块就扫一遍
        S.progress = true;
      })
      .catch((e) => {
        console.warn(e);
        setTimeout(() => (S.progress = true), 1000); // 过一会儿重画，缺的块会被重新请求
      })
      .finally(() => {
        if (gen === S.gen) S.pending.delete(chunkKey(cx, cy));
        S.inflight--;
        pump();
      });
  }
}

export async function loadWorld(params) {
  S.gen++;
  // d = 服务端数据版本（见 app.py），改了生成代码后让旧的缓存地址失效
  S.query = new URLSearchParams({ ...params, d: DATA_VERSION }).toString();
  S.chunks.clear();
  S.pending.clear();
  S.queue = [];
  S.bakeQueue = [];
  S.bakeKeys.clear();
  S.bitmaps.clear();
  S.bitmapPixels = 0;
  S.scaledBitmaps.clear();S.scaledPixels=0;
  S.hover = null;
  S.areaEdges = null;
  S.meta = null;
  S.dirty = true;
  const gen = S.gen;
  const meta = await fetch(`/api/world?${S.query}`).then((r) => {
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    return r.json();
  });
  if (gen !== S.gen) return null;
  S.meta = meta;
  S.W = meta.width;
  S.H = meta.height;
  S.CS = meta.chunk;
  S.PW = SQRT3 * S.W;
  buildLut(meta);
  S.rivers = meta.rivers.map((r) => {
    let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
    for (let k = 0; k < r.p.length; k += 2) {
      x0 = Math.min(x0, r.p[k]); x1 = Math.max(x1, r.p[k]);
      y0 = Math.min(y0, r.p[k + 1]); y1 = Math.max(y1, r.p[k + 1]);
    }
    return { w: r.w, p: r.p, box: [x0, y0, x1, y1] };
  });
  fitView();
  return meta;
}

// ---------------------------------------------------------------- 相机

function mapTop() {
  return -1.2;
}
function mapBottom() {
  return 1.5 * (S.H - 1) + 1.2;
}
function zoomLimits() {
  const h = mapBottom() - mapTop();
  const zMin = Math.max((S.vh / h) * 0.94, S.vw / S.PW);
  return [zMin, Math.max(zMin * 1.5, 128)];
}
function clampCam() {
  const [zMin, zMax] = zoomLimits();
  const c = S.cam;
  c.z = Math.min(zMax, Math.max(zMin, c.z));
  const half = S.vh / 2 / c.z;
  const top = mapTop() - 0.5;
  const bottom = mapBottom() + 0.5;
  c.y = bottom - top <= 2 * half ? (top + bottom) / 2 : Math.min(bottom - half, Math.max(top + half, c.y));
  c.x = ((c.x % S.PW) + S.PW) % S.PW;
}
export function fitView() {
  if (!S.meta) return;
  S.cam.z = 0;
  S.cam.x = S.PW / 2 - SQRT3 / 2;
  S.cam.y = (mapTop() + mapBottom()) / 2;
  clampCam();
  S.dirty = true;
}
export function zoomBy(f, sx = S.vw / 2, sy = S.vh / 2) {
  if (!S.meta) return;
  const c = S.cam;
  const wx = c.x + (sx - S.vw / 2) / c.z;
  const wy = c.y + (sy - S.vh / 2) / c.z;
  c.z *= f;
  clampCam();
  c.x = wx - (sx - S.vw / 2) / c.z;
  c.y = wy - (sy - S.vh / 2) / c.z;
  clampCam();
  S.dirty = true;
}
export function panBy(dx, dy) {
  S.cam.x -= dx / S.cam.z;
  S.cam.y -= dy / S.cam.z;
  clampCam();
  S.dirty = true;
}
function screenToWorld(sx, sy) {
  return [S.cam.x + (sx - S.vw / 2) / S.cam.z, S.cam.y + (sy - S.vh / 2) / S.cam.z];
}

function worldToTile(x, y) {
  const q = (SQRT3 / 3) * x - y / 3;
  const r = (2 / 3) * y;
  let rq = Math.round(q), rr = Math.round(r), rs = Math.round(-q - r);
  const dq = Math.abs(rq - q), dr = Math.abs(rr - r), ds = Math.abs(rs + q + r);
  if (dq > dr && dq > ds) rq = -rr - rs;
  else if (dr > ds) rr = -rq - rs;
  const col = rq + (rr - (rr & 1)) / 2;
  return [col, rr];
}

// ---------------------------------------------------------------- 绘制单个格子

function hexPath(ctx, x, y, r) {
  ctx.beginPath();
  ctx.moveTo(x + CORNERS[0][0] * r, y + CORNERS[0][1] * r);
  for (let i = 1; i < 6; i++) ctx.lineTo(x + CORNERS[i][0] * r, y + CORNERS[i][1] * r);
  ctx.closePath();
}

function edgeLine(ctx, x, y, s, d) {
  const [a, b] = EDGE_CORNERS[d];
  ctx.moveTo(x + CORNERS[a][0] * s, y + CORNERS[a][1] * s);
  ctx.lineTo(x + CORNERS[b][0] * s, y + CORNERS[b][1] * s);
}

function drawMountain(ctx, x, y, s, h) {
  const jit = (((h >>> 4) & 15) / 15 - 0.5) * 0.12 * s;
  const peaks = [
    [-0.24, 0.36, 0.5, 0.78, true],
    [0.26, 0.4, 0.4, 0.56, false],
  ];
  ctx.fillStyle = "rgba(30,24,18,0.16)";
  ctx.beginPath();
  ctx.ellipse(x, y + 0.42 * s, 0.72 * s, 0.13 * s, 0, 0, Math.PI * 2);
  ctx.fill();
  for (const [dx, by0, hw0, h0, cap] of peaks) {
    const bx = x + dx * s + jit;
    const by = y + by0 * s;
    const hw = hw0 * s;
    const ph = h0 * s;
    const ax = bx + hw * 0.08;
    const ay = by - ph;
    const mid = bx + hw * 0.18;
    ctx.fillStyle = "#c2b8aa";
    ctx.beginPath();
    ctx.moveTo(bx - hw, by); ctx.lineTo(ax, ay); ctx.lineTo(mid, by); ctx.closePath();
    ctx.fill();
    ctx.fillStyle = "#6e665e";
    ctx.beginPath();
    ctx.moveTo(ax, ay); ctx.lineTo(bx + hw, by); ctx.lineTo(mid, by); ctx.closePath();
    ctx.fill();
    if (cap) {
      ctx.fillStyle = "#f3f1ec";
      ctx.beginPath();
      ctx.moveTo(ax, ay);
      ctx.lineTo(ax - hw * 0.34, ay + ph * 0.3);
      ctx.lineTo(ax - hw * 0.08, ay + ph * 0.22);
      ctx.lineTo(ax + hw * 0.1, ay + ph * 0.3);
      ctx.lineTo(ax + hw * 0.26, ay + ph * 0.24);
      ctx.closePath();
      ctx.fill();
    }
  }
}

// 火山：截顶的深色锥体，火山口一点暗红，顶上一缕烟
function drawVolcano(ctx, x, y, s) {
  const hw = 0.5 * s, ph = 0.7 * s, top = 0.12 * s;
  const by = y + 0.36 * s, ty = by - ph;
  ctx.fillStyle = "rgba(30,24,18,0.18)";
  ctx.beginPath();
  ctx.ellipse(x, by + 0.04 * s, 0.68 * s, 0.12 * s, 0, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = "#8a7a6e";
  ctx.beginPath();
  ctx.moveTo(x - hw, by); ctx.lineTo(x - top, ty); ctx.lineTo(x + 0.02 * s, ty); ctx.lineTo(x + 0.05 * s, by); ctx.closePath();
  ctx.fill();
  ctx.fillStyle = "#4f4540";
  ctx.beginPath();
  ctx.moveTo(x + 0.02 * s, ty); ctx.lineTo(x + top, ty); ctx.lineTo(x + hw, by); ctx.lineTo(x + 0.05 * s, by); ctx.closePath();
  ctx.fill();
  ctx.fillStyle = "#b5432e";
  ctx.beginPath();
  ctx.ellipse(x, ty + 0.01 * s, top * 0.95, 0.035 * s + 0.5, 0, 0, Math.PI * 2);
  ctx.fill();
  if (s >= 10) {
    ctx.strokeStyle = "rgba(235,232,226,0.55)";
    ctx.lineWidth = Math.max(1, 0.05 * s);
    ctx.beginPath();
    ctx.moveTo(x, ty - 0.04 * s);
    ctx.bezierCurveTo(x - 0.12 * s, ty - 0.2 * s, x + 0.14 * s, ty - 0.28 * s, x + 0.04 * s, ty - 0.42 * s);
    ctx.stroke();
  }
}

function drawHills(ctx, x, y, s, h) {
  const jit = (((h >>> 5) & 15) / 15 - 0.5) * 0.1 * s;
  for (const [dx, dy, w] of [[-0.24, 0.2, 0.4], [0.24, 0.32, 0.34]]) {
    const bx = x + dx * s + jit;
    const by = y + dy * s;
    const hw = w * s;
    ctx.fillStyle = "rgba(108,86,56,0.38)";
    ctx.beginPath();
    ctx.moveTo(bx - hw, by);
    ctx.quadraticCurveTo(bx, by - hw * 1.05, bx + hw, by);
    ctx.closePath();
    ctx.fill();
    ctx.strokeStyle = "rgba(250,236,200,0.45)";
    ctx.lineWidth = Math.max(0.6, s * 0.035);
    ctx.beginPath();
    ctx.moveTo(bx - hw * 0.85, by - hw * 0.08);
    ctx.quadraticCurveTo(bx - hw * 0.45, by - hw * 0.5, bx - hw * 0.02, by - hw * 0.52);
    ctx.stroke();
  }
}

function drawTrees(ctx, x, y, s, h, dark) {
  const spots = [[-0.32, 0.12], [0.02, -0.22], [0.32, 0.06], [-0.04, 0.34], [0.36, 0.42]];
  const n = 3 + (h % 3);
  const crown = dark ? "#2f4a2c" : "#405c33";
  for (let k = 0; k < n; k++) {
    const [dx, dy] = spots[k];
    const jx = ((((h >>> (k * 3)) & 7) / 7) - 0.5) * 0.1 * s;
    const tx = x + dx * s + jx;
    const ty = y + dy * s;
    const r = s * (0.15 + ((h >>> (k + 7)) & 3) * 0.012);
    ctx.fillStyle = "rgba(20,30,15,0.22)";
    ctx.beginPath();
    ctx.ellipse(tx + r * 0.25, ty + r * 0.75, r * 0.9, r * 0.35, 0, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = crown;
    ctx.beginPath();
    ctx.arc(tx, ty, r, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = "rgba(190,215,150,0.28)";
    ctx.beginPath();
    ctx.arc(tx - r * 0.3, ty - r * 0.32, r * 0.42, 0, Math.PI * 2);
    ctx.fill();
  }
}

function drawMarsh(ctx, x, y, s) {
  ctx.strokeStyle = "rgba(58,88,86,0.5)";
  ctx.lineWidth = Math.max(0.6, s * 0.045);
  ctx.beginPath();
  for (const [dx, dy, w] of [[-0.3, -0.05, 0.28], [0.05, 0.2, 0.32], [0.25, -0.2, 0.22]]) {
    ctx.moveTo(x + (dx - w / 2) * s, y + dy * s);
    ctx.lineTo(x + (dx + w / 2) * s, y + dy * s);
  }
  ctx.stroke();
}

function drawDunes(ctx, x, y, s) {
  ctx.strokeStyle = "rgba(150,118,72,0.32)";
  ctx.lineWidth = Math.max(0.6, s * 0.04);
  ctx.beginPath();
  ctx.moveTo(x - 0.4 * s, y + 0.15 * s);
  ctx.quadraticCurveTo(x - 0.1 * s, y - 0.12 * s, x + 0.2 * s, y + 0.12 * s);
  ctx.moveTo(x - 0.05 * s, y + 0.42 * s);
  ctx.quadraticCurveTo(x + 0.2 * s, y + 0.22 * s, x + 0.42 * s, y + 0.4 * s);
  ctx.stroke();
}

// tiles: [{x, y, ch, li, idx}]，x / y 为当前画布上的像素坐标；s = 格子半径（像素）
function drawFills(ctx, tiles, s) {
  for (const t of tiles) {
    ctx.fillStyle = fillColor(t);
    hexPath(ctx, t.x, t.y, s * 1.04);
    ctx.fill();
  }
}

function drawTiles(ctx, tiles, s, layers) {
  drawFills(ctx, tiles, s);
  drawDetails(ctx, tiles, s, layers);
}

function drawDetails(ctx, tiles, s, layers) {
  const R = S.roles;
  tiles = tiles.filter((t) => !t.ghost);
  const terrainMode = S.mode === "terrain";
  const areaMode = S.mode === "areas";
  if (terrainMode && s < DETAIL.glyphs && s >= 2.5) {
    // 远景也标出火山（缩小后其他地形图标不画）
    for (const t of tiles) {
      if (t.ch.terrain[t.li] === R.volcano) drawVolcano(ctx, t.x, t.y, Math.max(s, 5));
    }
  }
  if (terrainMode && s >= DETAIL.glyphs) {
    for (const t of tiles) {
      const ti = t.ch.terrain[t.li];
      const h = hash(t.idx);
      if (ti === R.volcano) drawVolcano(ctx, t.x, t.y, s);
      else if (ti === R.mountains) drawMountain(ctx, t.x, t.y, s, h);
      else if (ti === R.hills) drawHills(ctx, t.x, t.y, s, h);
      else if (ti === R.forest) drawTrees(ctx, t.x, t.y, s, h, false);
      else if (ti === R.rainforest) drawTrees(ctx, t.x, t.y, s, h, true);
      else if (ti === R.marsh) drawMarsh(ctx, t.x, t.y, s);
      else if (ti === R.desert && s >= 16 && h % 3 === 0) drawDunes(ctx, t.x, t.y, s);
    }
  }

  if (layers.grid && s >= DETAIL.grid) {
    ctx.strokeStyle = "rgba(20,20,20,0.08)";
    ctx.lineWidth = 1;
    for (const t of tiles) {
      hexPath(ctx, t.x, t.y, s);
      ctx.stroke();
    }
  }

  // "国家"模式不画大区边界（虚线留给城市辖区分界）
  if (areaMode || (layers.areas && s >= DETAIL.borders && S.mode !== "nations")) {
    ctx.strokeStyle = areaMode ? "rgba(40,32,24,0.45)" : "rgba(64,50,32,0.3)";
    ctx.lineWidth = Math.max(areaMode ? 0.6 : 0.7, s * 0.05);
    ctx.setLineDash(areaMode ? [] : [Math.max(2, s * 0.22), Math.max(1.5, s * 0.14)]);
    ctx.beginPath();
    for (const t of tiles) {
      const b = t.ch.aborder[t.li];
      if (!b || S.meta.ruleset.terrain[t.ch.terrain[t.li]].water) continue;
      for (const d of [0, 4, 5]) if (b & (1 << d)) edgeLine(ctx, t.x, t.y, s, d);
    }
    ctx.stroke();
    ctx.setLineDash([]);
  }

  if (s >= 8) {
    // 海岸浪花：水域一侧一圈浅色
    ctx.strokeStyle = "rgba(232,242,246,0.32)";
    ctx.lineWidth = s * 0.16;
    ctx.lineCap = "round";
    ctx.beginPath();
    for (const t of tiles) {
      const b = t.ch.coast[t.li];
      if (!b || !S.meta.ruleset.terrain[t.ch.terrain[t.li]].water) continue;
      for (let d = 0; d < 6; d++) if (b & (1 << d)) edgeLine(ctx, t.x, t.y, s, d);
    }
    ctx.stroke();
  }

  if (s >= DETAIL.coast) {
    ctx.strokeStyle = "rgba(34,40,46,0.55)";
    ctx.lineWidth = Math.max(0.8, s * 0.08);
    ctx.lineCap = "round";
    ctx.beginPath();
    for (const t of tiles) {
      const b = t.ch.coast[t.li];
      if (!b || S.meta.ruleset.terrain[t.ch.terrain[t.li]].water) continue;
      for (let d = 0; d < 6; d++) if (b & (1 << d)) edgeLine(ctx, t.x, t.y, s, d);
    }
    ctx.stroke();
  }

  if (layers.resources && s >= DETAIL.resources) {
    const res = S.meta.ruleset.resources;
    for (const t of tiles) {
      const r = t.ch.resource[t.li];
      if (!r) continue;
      const rr = Math.max(2.2, s * 0.17);
      const cx = t.x + s * 0.42;
      const cy = t.y - s * 0.38;
      ctx.fillStyle = "rgba(255,252,244,0.92)";
      ctx.beginPath();
      ctx.arc(cx, cy, rr + Math.max(1, s * 0.04), 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = res[r - 1].color;
      ctx.beginPath();
      ctx.arc(cx, cy, rr, 0, Math.PI * 2);
      ctx.fill();
    }
  }
}

// ---------------------------------------------------------------- 分块位图

function chunkBounds(ch) {
  return [
    SQRT3 * (ch.c0 - RING) - SQRT3,
    1.5 * (ch.r0 - RING) - 1.3,
    SQRT3 * (ch.c0 + ch.cw + RING) + SQRT3,
    1.5 * (ch.r0 + ch.ch - 1 + RING) + 1.3,
  ];
}

// 分块真正负责显示的区域（世界坐标）；相邻分块的裁剪区正好无缝拼接
function chunkCrop(ch) {
  return [
    SQRT3 * (ch.c0 - 0.25),
    ch.r0 === 0 ? -1.3 : 1.5 * ch.r0 - 0.75,
    SQRT3 * (ch.c0 + ch.cw - 0.25),
    ch.r0 + ch.ch >= S.H ? 1.5 * (S.H - 1) + 1.3 : 1.5 * (ch.r0 + ch.ch) - 0.75,
  ];
}

// 分块及外圈 RING 格（外圈数据来自邻块，邻块没加载时标记为不完整）
function chunkTiles(ch, s, bx0, by0) {
  const out = [];
  let complete = true;
  for (let row = ch.r0 - RING; row < ch.r0 + ch.ch + RING; row++) {
    // 地图上下边缘之外用边缘行的数据补一圈"影子格"，只参与柔化，不画细节
    const src = Math.max(0, Math.min(S.H - 1, row));
    for (let ucol = ch.c0 - RING; ucol < ch.c0 + ch.cw + RING; ucol++) {
      const td = tileData(ucol, src);
      if (!td) {
        complete = false;
        const col = ((ucol % S.W) + S.W) % S.W;
        requestChunk((col / S.CS) | 0, (src / S.CS) | 0);
        continue;
      }
      out.push({
        x: (SQRT3 * (ucol + 0.5 * (row & 1)) - bx0) * s,
        y: (1.5 * row - by0) * s,
        ch: td.ch,
        li: td.li,
        idx: src * S.W + td.col,
        ghost: row !== src,
      });
    }
  }
  return { tiles: out, complete };
}

function layerKey() {
  const l = S.layers;
  return `${S.mode}${+l.areas}${+l.resources}${+l.grid}`;
}

let scratch = null;

function getBitmap(ch, tier) {
  const key = `${ch.cx},${ch.cy},${tier},${layerKey()}`;
  const bm = S.bitmaps.get(key);
  if (bm) {
    S.bitmaps.delete(key);
    S.bitmaps.set(key, bm); // LRU
    bm.usedAt = S.frameNo;
  }
  return bm;
}

// 当前层级的位图还没烤好时，先拿别的层级已有的顶上（缩放跨层级时不出现空洞）
function fallbackBitmap(ch, tier) {
  const lk = layerKey();
  const order = TIERS.filter((t) => t !== tier).sort((a, b) => Math.abs(a - tier) - Math.abs(b - tier));
  for (const t of order) {
    const bm = S.bitmaps.get(`${ch.cx},${ch.cy},${t},${lk}`);
    if (bm) {
      bm.usedAt = S.frameNo;
      return bm;
    }
  }
  return null;
}

function queueBake(ch, tier) {
  const key = `${ch.cx},${ch.cy},${tier}`;
  if (S.bakeKeys.has(key)) return;
  S.bakeKeys.add(key);
  S.bakeQueue.push({ ch, tier });
}

// 位图要用到四周一圈邻块的数据：邻块都到了再烤，免得先烤一张残缺的、邻块一到又得重烤
function neighborsReady(ch) {
  const ncx = Math.ceil(S.W / S.CS), ncy = Math.ceil(S.H / S.CS);
  let ready = true;
  for (let cy = Math.max(0, ch.cy - 1); cy <= Math.min(ncy - 1, ch.cy + 1); cy++) {
    for (let dx = -1; dx <= 1; dx++) {
      const cx = (ch.cx + dx + ncx) % ncx;
      if (!S.chunks.has(chunkKey(cx, cy))) {
        requestChunk(cx, cy);
        ready = false;
      }
    }
  }
  return ready;
}

function bakeBitmap(ch, tier) {
  const key = `${ch.cx},${ch.cy},${tier},${layerKey()}`;
  const [bx0, by0, bx1, by1] = chunkBounds(ch);
  const w = Math.ceil((bx1 - bx0) * tier);
  const h = Math.ceil((by1 - by0) * tier);
  const { tiles, complete } = chunkTiles(ch, tier, bx0, by0);

  // 先把地形色块画在草稿上，柔化后贴到位图，让相邻格子自然过渡（远景像一张整图）
  scratch ??= document.createElement("canvas");
  scratch.width = w;
  scratch.height = h;
  const sctx = scratch.getContext("2d",{willReadFrequently:true});
  drawFills(sctx, tiles, tier);
  const cv = document.createElement("canvas");
  cv.width = w;
  cv.height = h;
  const ctx = cv.getContext("2d",{willReadFrequently:true});
  // 只有远景做明显柔化（整张图的颜色过渡）；中近景只轻微柔化，保持清晰
  const blur = tier <= 6 ? Math.max(0.6, tier * 0.28) : tier <= 12 ? 1.0 : 0.5;
  ctx.filter = `blur(${blur.toFixed(2)}px)`;
  ctx.drawImage(scratch, 0, 0);
  ctx.filter = "none";
  drawDetails(ctx, tiles, tier, S.layers);

  const bm = { canvas: cv, bounds: [bx0, by0, bx1, by1], cx: ch.cx, cy: ch.cy, tier, complete, pixels: w * h };
  S.bitmaps.set(key, bm);
  S.bitmapPixels += bm.pixels;
  // 按最近使用淘汰；本帧正在用的不淘汰（Map 的顺序就是使用顺序）
  for (const [k, old] of S.bitmaps) {
    if (S.bitmapPixels <= MAX_BITMAP_PIXELS || old.usedAt === S.frameNo) break;
    S.bitmaps.delete(k);
    S.bitmapPixels -= old.pixels;
  }
  bm.usedAt = S.frameNo;
  return bm;
}

// 新分块到达后，丢掉旁边那些当时缺邻块数据的位图，下一帧重画
function invalidateNeighbors(cx, cy) {
  const ncx = Math.ceil(S.W / S.CS);
  for (const [key, bm] of S.bitmaps) {
    if (bm.complete) continue;
    const dx = Math.min(Math.abs(bm.cx - cx), ncx - Math.abs(bm.cx - cx));
    if (dx <= 1 && Math.abs(bm.cy - cy) <= 1) {
      S.bitmaps.delete(key);
      S.bitmapPixels -= bm.pixels;
    }
  }
}

// ---------------------------------------------------------------- 主绘制

function visibleRange() {
  const c = S.cam;
  const x0 = c.x - S.vw / 2 / c.z, x1 = c.x + S.vw / 2 / c.z;
  const y0 = c.y - S.vh / 2 / c.z, y1 = c.y + S.vh / 2 / c.z;
  return { x0, x1, y0, y1 };
}

function drawBitmaps(ctx, vis, phys) {
  const tier = TIERS.find((t) => t >= phys * 0.9) ?? TIERS[TIERS.length - 1];
  const c = S.cam;
  let missing = false;
  const ncx = Math.ceil(S.W / S.CS), ncy = Math.ceil(S.H / S.CS);
  const cy0 = Math.max(0, Math.floor((vis.y0 - 1.5) / 1.5 / S.CS));
  const cy1 = Math.min(ncy - 1, Math.floor((vis.y1 + 1.5) / 1.5 / S.CS));
  const k0 = Math.floor((vis.x0 + SQRT3) / S.PW) - 1, k1 = Math.floor((vis.x1 + SQRT3) / S.PW) + 1;
  for (let k = k0; k <= k1; k++) {
    for (let cy = cy0; cy <= cy1; cy++) {
      for (let cx = 0; cx < ncx; cx++) {
        const left = SQRT3 * cx * S.CS - SQRT3 + k * S.PW;
        const right = SQRT3 * (cx + 1) * S.CS + SQRT3 + k * S.PW;
        if (right < vis.x0 || left > vis.x1) continue;
        const ch = S.chunks.get(chunkKey(cx, cy));
        if (!ch) {
          requestChunk(cx, cy);
          missing = true;
          continue;
        }
        let bm = getBitmap(ch, tier);
        if (!bm) {
          missing = true;
          queueBake(ch, tier);
          bm = fallbackBitmap(ch, tier);
          if (!bm) continue;
        }
        const [x0, y0, x1, y1] = chunkCrop(ch);
        const pad = 0.6 / c.z; // 多盖半个像素，避免缩放后块缝露底
        const bitmap=scaledBitmap(bm,phys,[x0,y0,x1,y1]);
        ctx.drawImage(
          bitmap,
          (x0 + k * S.PW - c.x) * c.z + S.vw / 2,
          (y0 - c.y) * c.z + S.vh / 2,
          (x1 - x0 + pad) * c.z,
          (y1 - y0 + pad) * c.z,
        );
      }
    }
  }
  return missing;
}

let bitmapSerial=0;
function scaledBitmap(bm,phys,crop) {
  bm.serial??=++bitmapSerial;
  const key=`${bm.serial}:${phys.toFixed(4)}`;
  let cv=S.scaledBitmaps.get(key);
  if(cv){S.scaledBitmaps.delete(key);S.scaledBitmaps.set(key,cv);return cv;}
  const [x0,y0,x1,y1]=crop,[bx0,by0]=bm.bounds,bt=bm.tier;
  cv=document.createElement('canvas');
  cv.width=Math.ceil((x1-x0)*phys);cv.height=Math.ceil((y1-y0)*phys);
  const ctx=cv.getContext('2d',{willReadFrequently:true});ctx.imageSmoothingQuality='high';
  ctx.drawImage(bm.canvas,(x0-bx0)*bt,(y0-by0)*bt,(x1-x0)*bt,(y1-y0)*bt,0,0,cv.width,cv.height);
  S.scaledBitmaps.set(key,cv);S.scaledPixels+=cv.width*cv.height;
  while(S.scaledPixels>8_000_000&&S.scaledBitmaps.size>1) {
    const oldest=S.scaledBitmaps.keys().next().value,old=S.scaledBitmaps.get(oldest);
    S.scaledPixels-=old.width*old.height;S.scaledBitmaps.delete(oldest);
  }
  return cv;
}

function drawVector(ctx, vis) {
  const c = S.cam;
  const r0 = Math.max(0, Math.floor((vis.y0 - 1) / 1.5));
  const r1 = Math.min(S.H - 1, Math.ceil((vis.y1 + 1) / 1.5));
  const c0 = Math.floor(vis.x0 / SQRT3) - 1, c1 = Math.ceil(vis.x1 / SQRT3) + 1;
  const tiles = [];
  let missing = false;
  for (let row = r0; row <= r1; row++) {
    for (let ucol = c0; ucol <= c1; ucol++) {
      const td = tileData(ucol, row);
      if (!td) {
        const col = ((ucol % S.W) + S.W) % S.W;
        requestChunk((col / S.CS) | 0, (row / S.CS) | 0);
        missing = true;
        continue;
      }
      tiles.push({
        x: (SQRT3 * (ucol + 0.5 * (row & 1)) - c.x) * c.z + S.vw / 2,
        y: (1.5 * row - c.y) * c.z + S.vh / 2,
        ch: td.ch,
        li: td.li,
        idx: row * S.W + td.col,
      });
    }
  }
  drawFills(ctx, tiles, c.z);
  if (S.mode === "terrain") drawTextures(ctx, tiles, c.z);
  drawDetails(ctx, tiles, c.z, S.layers);
  return missing;
}

// ---------------------------------------------------------------- 近景细节贴图

// 每种地形一张 128 px 的可平铺纹理（只画半透明的明暗细节，底色仍由地形色决定）
const textures = new Map();
function makeRng(seed) {
  let x = seed >>> 0 || 1;
  return () => ((x = Math.imul(x ^ (x >>> 15), 0x2c1b3c6d) + 0x9e3779b9) >>> 0) / 4294967296;
}
function textureFor(ctx, ti) {
  if (textures.has(ti)) return textures.get(ti);
  const R = S.roles;
  const size = 128;
  const cv = document.createElement("canvas");
  cv.width = cv.height = size;
  const g = cv.getContext("2d");
  const rnd = makeRng(ti * 7919 + 17);
  const light = (a) => `rgba(255,250,235,${a})`;
  const dark = (a) => `rgba(30,24,16,${a})`;
  const wrap = (fn) => {
    // 画在 3×3 的位置上，保证纹理左右上下无缝
    for (const ox of [-size, 0, size]) for (const oy of [-size, 0, size]) fn(ox, oy);
  };
  const dots = (count, rMin, rMax, color) => {
    for (let i = 0; i < count; i++) {
      const x = rnd() * size, y = rnd() * size, r = rMin + rnd() * (rMax - rMin);
      g.fillStyle = typeof color === "function" ? color() : color;
      wrap((ox, oy) => { g.beginPath(); g.arc(x + ox, y + oy, r, 0, Math.PI * 2); g.fill(); });
    }
  };
  const strokes = (count, len, color, width, slant) => {
    g.lineCap = "round";
    for (let i = 0; i < count; i++) {
      const x = rnd() * size, y = rnd() * size, a = slant + (rnd() - 0.5) * 0.6;
      g.strokeStyle = typeof color === "function" ? color() : color;
      g.lineWidth = width;
      wrap((ox, oy) => {
        g.beginPath();
        g.moveTo(x + ox, y + oy);
        g.lineTo(x + ox + Math.cos(a) * len, y + oy + Math.sin(a) * len);
        g.stroke();
      });
    }
  };
  if (ti === R.grassland || ti === R.plains) {
    strokes(260, 3.2, () => (rnd() < 0.5 ? light(0.16) : dark(0.12)), 1, -Math.PI / 2.4);
    dots(40, 1.5, 4, () => dark(0.05));
  } else if (ti === R.desert) {
    g.lineWidth = 1.1;
    for (let k = 0; k < 9; k++) {
      const y0 = (k + rnd() * 0.6) * (size / 9), amp = 2 + rnd() * 3, ph = rnd() * Math.PI * 2;
      for (const [col, off] of [[light(0.2), 0], [dark(0.12), 1.6]]) {
        g.strokeStyle = col;
        g.beginPath();
        for (let x = 0; x <= size; x += 4) {
          const y = y0 + off + Math.sin((x / size) * Math.PI * 4 + ph) * amp;
          x === 0 ? g.moveTo(x, y) : g.lineTo(x, y);
        }
        g.stroke();
      }
    }
    dots(60, 0.6, 1.2, () => dark(0.1));
  } else if (ti === R.forest || ti === R.rainforest) {
    dots(70, 3, 7, () => dark(0.09));
    dots(90, 1, 2.5, () => light(0.08));
  } else if (ti === R.tundra || ti === R.marsh) {
    dots(220, 0.7, 1.8, () => (rnd() < 0.5 ? light(0.14) : dark(0.1)));
    if (ti === R.marsh) strokes(40, 7, "rgba(40,80,80,0.18)", 1.2, 0);
  } else if (ti === R.snow || ti === R.ice) {
    dots(160, 0.6, 1.6, () => "rgba(120,150,175,0.07)");
  } else if (ti === R.hills || ti === R.mountains || ti === R.volcano) {
    dots(140, 0.8, 2.6, () => (rnd() < 0.5 ? dark(0.12) : light(0.12)));
    strokes(50, 5, () => dark(0.08), 1, 0.9);
  } else {
    // 水面：浅浅的波纹
    g.lineWidth = 1;
    for (let i = 0; i < 46; i++) {
      const x = rnd() * size, y = rnd() * size, w = 5 + rnd() * 8;
      g.strokeStyle = `rgba(235,245,250,${0.05 + rnd() * 0.06})`;
      wrap((ox, oy) => {
        g.beginPath();
        g.moveTo(x + ox - w, y + oy);
        g.quadraticCurveTo(x + ox, y + oy - 2.2, x + ox + w, y + oy);
        g.stroke();
      });
    }
  }
  const pat = ctx.createPattern(cv, "repeat");
  textures.set(ti, pat);
  return pat;
}

function drawTextures(ctx, tiles, s) {
  // 纹理跟着世界坐标走：一张纹理覆盖 2 个世界单位，拖动地图时不会"漂"
  const c = S.cam;
  const k = (2 * c.z) / 128;
  const m = new DOMMatrix([k, 0, 0, k, -c.x * c.z + S.vw / 2, -c.y * c.z + S.vh / 2]);
  const byTerrain = new Map();
  for (const t of tiles) {
    const ti = t.ch.terrain[t.li];
    if (!byTerrain.has(ti)) byTerrain.set(ti, []);
    byTerrain.get(ti).push(t);
  }
  for (const [ti, list] of byTerrain) {
    const pat = textureFor(ctx, ti);
    pat.setTransform(m);
    ctx.fillStyle = pat;
    ctx.beginPath();
    for (const t of list) {
      ctx.moveTo(t.x + CORNERS[0][0] * s * 1.04, t.y + CORNERS[0][1] * s * 1.04);
      for (let i = 1; i < 6; i++) ctx.lineTo(t.x + CORNERS[i][0] * s * 1.04, t.y + CORNERS[i][1] * s * 1.04);
      ctx.closePath();
    }
    ctx.fill();
  }
}

function smoothPath(ctx, p, ox, oy, z) {
  const X = (k) => (p[k] + ox) * z;
  const Y = (k) => (p[k + 1] + oy) * z;
  ctx.moveTo(X(0), Y(0));
  if (p.length === 4) {
    ctx.lineTo(X(2), Y(2));
    return;
  }
  for (let k = 2; k < p.length - 2; k += 2) {
    ctx.quadraticCurveTo(X(k), Y(k), (X(k) + X(k + 2)) / 2, (Y(k) + Y(k + 2)) / 2);
  }
  ctx.lineTo(X(p.length - 2), Y(p.length - 2));
}

function drawRivers(ctx, vis) {
  const c = S.cam;
  // w = 流量（成河阈值 = 1，源头段 < 1）；缩小时只画大河
  const minFlow = c.z < 2.5 ? 4 : c.z < 5 ? 1.5 : 0;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.strokeStyle = "#3b6f99";
  const k0 = Math.floor(vis.x0 / S.PW) - 1, k1 = Math.floor(vis.x1 / S.PW) + 1;
  const groups = new Map();
  for (const r of S.rivers) {
    if (r.w < minFlow) continue;
    const width = Math.min(0.42 * c.z, 0.8 + c.z * 0.045 * Math.pow(r.w, 0.45));
    const key = Math.round(width * 4) / 4;
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(r);
  }
  for (const [width, rivers] of groups) {
    ctx.lineWidth = width;
    ctx.beginPath();
    for (let k = k0; k <= k1; k++) {
      const off = k * S.PW;
      for (const r of rivers) {
        const [x0, y0, x1, y1] = r.box;
        if (x1 + off < vis.x0 - 1 || x0 + off > vis.x1 + 1 || y1 < vis.y0 - 1 || y0 > vis.y1 + 1) continue;
        smoothPath(ctx, r.p, off - c.x + S.vw / 2 / c.z, -c.y + S.vh / 2 / c.z, c.z);
      }
    }
    ctx.stroke();
  }
}

function tileCenterScreen(col, row, k) {
  const c = S.cam;
  return [
    (SQRT3 * (col + 0.5 * (row & 1)) + k * S.PW - c.x) * c.z + S.vw / 2,
    (1.5 * row - c.y) * c.z + S.vh / 2,
  ];
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

function drawStarts(ctx, vis) {
  const z = S.cam.z;
  const k0 = Math.floor(vis.x0 / S.PW) - 1, k1 = Math.floor(vis.x1 / S.PW) + 1;
  ctx.font = `600 ${Math.round(Math.min(13, 9 + z * 0.15))}px -apple-system, "PingFang SC", sans-serif`;
  ctx.textBaseline = "middle";
  S.meta.starts.forEach((st, n) => {
    for (let k = k0; k <= k1; k++) {
      const [x, y] = tileCenterScreen(st.col, st.row, k);
      if (x < -20 || x > S.vw + 20 || y < -20 || y > S.vh + 20) continue;
      const r = Math.min(10, 4.5 + z * 0.12);
      ctx.shadowColor = "rgba(0,0,0,0.45)";
      ctx.shadowBlur = 4;
      drawStar(ctx, x, y, r);
      ctx.fillStyle = "#fbf6e8";
      ctx.fill();
      ctx.shadowBlur = 0;
      ctx.lineWidth = 1.2;
      ctx.strokeStyle = "#5a3d1e";
      ctx.stroke();
      if (z >= 5) {
        ctx.fillStyle = "#fbf6e8";
        ctx.strokeStyle = "rgba(20,16,10,0.75)";
        ctx.lineWidth = 3;
        ctx.strokeText(String(n + 1), x + r + 3, y);
        ctx.fillText(String(n + 1), x + r + 3, y);
      }
    }
  });
}

function computeAreaEdges(area) {
  // 在已加载的分块里找出该大区的所有格子，记下它们的外边
  const edges = [];
  for (const ch of S.chunks.values()) {
    for (let li = 0; li < ch.area.length; li++) {
      if (ch.area[li] !== area) continue;
      const mask = ch.aborder[li] | ch.coast[li];
      const row = ch.r0 + ((li / ch.cw) | 0);
      const col = ch.c0 + (li % ch.cw);
      for (let d = 0; d < 6; d++) {
        if (mask & (1 << d) || (row === 0 && (d === 1 || d === 2)) || (row === S.H - 1 && (d === 4 || d === 5))) {
          edges.push([col, row, d]);
        }
      }
    }
  }
  S.areaEdges = { area, edges };
}

function drawHover(ctx, vis) {
  if (!S.hover || !S.areaEdges || S.cam.z < 2 || S.mode === "nations") return;
  if (S.edgesStale) {
    S.edgesStale = false;
    computeAreaEdges(S.hover.area);
  }
  const z = S.cam.z;
  const k0 = Math.floor(vis.x0 / S.PW) - 1, k1 = Math.floor(vis.x1 / S.PW) + 1;
  ctx.lineCap = "round";
  ctx.beginPath();
  for (let k = k0; k <= k1; k++) {
    for (const [col, row, d] of S.areaEdges.edges) {
      const [x, y] = tileCenterScreen(col, row, k);
      if (x < -z * 2 || x > S.vw + z * 2 || y < -z * 2 || y > S.vh + z * 2) continue;
      edgeLine(ctx, x, y, z, d);
    }
  }
  ctx.strokeStyle = "rgba(20,14,6,0.45)";
  ctx.lineWidth = 4;
  ctx.stroke();
  ctx.strokeStyle = "rgba(255,246,222,0.95)";
  ctx.lineWidth = 1.8;
  ctx.stroke();
}

function arrow(ctx, x, y, dx, dy, len, head) {
  const n = Math.hypot(dx, dy) || 1;
  const ux = dx / n, uy = dy / n;
  const x0 = x - (ux * len) / 2, y0 = y - (uy * len) / 2;
  const x1 = x + (ux * len) / 2, y1 = y + (uy * len) / 2;
  ctx.moveTo(x0, y0);
  ctx.lineTo(x1, y1);
  ctx.moveTo(x1 - ux * head - uy * head * 0.6, y1 - uy * head + ux * head * 0.6);
  ctx.lineTo(x1, y1);
  ctx.lineTo(x1 - ux * head + uy * head * 0.6, y1 - uy * head - ux * head * 0.6);
}

function strokeTwice(ctx, light, width) {
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.strokeStyle = "rgba(10,12,16,0.55)";
  ctx.lineWidth = width + 2.5;
  ctx.stroke();
  ctx.strokeStyle = light;
  ctx.lineWidth = width;
  ctx.stroke();
}

// 地面风箭头：屏幕上每隔约 70 像素取一格（放大到一格比这还大时每格一个），长度表示风速
function drawWind(ctx) {
  const tilePx = SQRT3 * S.cam.z;
  const step = Math.max(70, tilePx);
  ctx.beginPath();
  for (let sy = step / 2, k = 0; sy < S.vh; sy += step * 0.866, k++) {
    for (let sx = step / 2 + (k % 2) * (step / 2); sx < S.vw; sx += step) {
      const [wx, wy] = screenToWorld(sx, sy);
      const [col, row] = worldToTile(wx, wy);
      const td = tileData(col, row);
      if (!td) continue;
      const [u, v, speed] = windOf(td);
      if (speed < 0.05) continue;
      const len = Math.min(step * 0.8, 10 + 34 * Math.min(1.2, speed));
      arrow(ctx, sx, sy, u, v, len, 4 + 3 * Math.min(1, speed));
    }
  }
  strokeTwice(ctx, "rgba(250,248,240,0.85)", 1.6);
}

// 洋流箭头：只画在海上，红 = 暖流、蓝 = 寒流、浅色 = 海温正常；长度表示流速
function currentOf(td) {
  const a = (td.ch.cdir[td.li] / 256) * 2 * Math.PI;
  const speed = td.ch.cspd[td.li] / S.meta.wind_speed_scale;
  return [Math.cos(a) * speed, Math.sin(a) * speed, speed, td.ch.canom[td.li] / 10];
}

function drawCurrents(ctx) {
  const tilePx = SQRT3 * S.cam.z;
  const step = Math.max(46, tilePx);
  const groups = { warm: new Path2D(), cold: new Path2D(), calm: new Path2D() };
  for (let sy = step / 2, k = 0; sy < S.vh; sy += step * 0.866, k++) {
    for (let sx = step / 2 + (k % 2) * (step / 2); sx < S.vw; sx += step) {
      const [wx, wy] = screenToWorld(sx, sy);
      const [col, row] = worldToTile(wx, wy);
      const td = tileData(col, row);
      if (!td || !isWater(td.ch.terrain[td.li])) continue;
      const [u, v, speed, anom] = currentOf(td);
      if (speed < 0.03) continue;
      const g = anom > 1 ? groups.warm : anom < -1 ? groups.cold : groups.calm;
      const len = Math.min(step * 0.85, 8 + 40 * Math.min(1, speed / 0.8));
      const p = new Path2D();
      arrow(p, sx, sy, u, v, len, 3.5 + 2 * Math.min(1, speed / 0.8));
      g.addPath(p);
    }
  }
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  for (const [key, color] of [["calm", "rgba(225,236,242,0.75)"], ["cold", "#6fb6e6"], ["warm", "#ef8a5c"]]) {
    ctx.strokeStyle = "rgba(10,12,16,0.5)";
    ctx.lineWidth = 3.6;
    ctx.stroke(groups[key]);
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.6;
    ctx.stroke(groups[key]);
  }
}

// 板块运动箭头：画在每个板块的几何中心，长度表示速度
function drawPlateArrows(ctx, vis) {
  const k0 = Math.floor(vis.x0 / S.PW) - 1, k1 = Math.floor(vis.x1 / S.PW) + 1;
  ctx.beginPath();
  for (const p of S.meta.plates) {
    for (let k = k0; k <= k1; k++) {
      const [x, y] = tileCenterScreen(Math.round(p.col), Math.round(p.row), k);
      if (x < -60 || x > S.vw + 60 || y < -60 || y > S.vh + 60) continue;
      arrow(ctx, x, y, p.vx, p.vy, 30 + 50 * Math.hypot(p.vx, p.vy), 9);
    }
  }
  strokeTwice(ctx, "rgba(255,250,236,0.95)", 2.4);
}

let grain = null;
function grainPattern(ctx) {
  if (grain) return grain;
  const cv = document.createElement("canvas");
  cv.width = cv.height = 192;
  const g = cv.getContext("2d");
  const img = g.createImageData(192, 192);
  for (let i = 0; i < img.data.length; i += 4) {
    const v = 128 + (Math.random() - 0.5) * 70;
    img.data[i] = img.data[i + 1] = img.data[i + 2] = v;
    img.data[i + 3] = 255;
  }
  g.putImageData(img, 0, 0);
  grain = ctx.createPattern(cv, "repeat");
  return grain;
}

function render() {
  S.frameNo++;
  S.bakeQueue = [];
  S.bakeKeys.clear();
  const ctx = S.ctx;
  ctx.setTransform(S.dpr, 0, 0, S.dpr, 0, 0);
  ctx.fillStyle = BACKGROUND;
  ctx.fillRect(0, 0, S.vw, S.vh);
  if (!S.meta) return false;
  const vis = visibleRange();
  const phys = S.cam.z * S.dpr;
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "low";
  const missing = phys > VECTOR_FROM ? drawVector(ctx, vis) : drawBitmaps(ctx, vis, phys);
  const view = overlayView(vis);
  if (S.overlay?.under) S.overlay.under(ctx, view);
  if (S.layers.rivers) drawRivers(ctx, vis);
  if (S.mode === "plates") drawPlateArrows(ctx, vis);
  if (S.layers.currents) drawCurrents(ctx);
  if (S.layers.wind || S.mode === "wind") drawWind(ctx);
  drawHover(ctx, vis);
  if (S.layers.starts) drawStarts(ctx, vis);
  if (S.overlay?.over) S.overlay.over(ctx, view);

  // 纸面颗粒 + 暗角，让画面更柔和
  ctx.save();
  ctx.globalCompositeOperation = "soft-light";
  ctx.globalAlpha = 0.22;
  ctx.fillStyle = grainPattern(ctx);
  ctx.fillRect(0, 0, S.vw, S.vh);
  ctx.restore();
  const g = ctx.createRadialGradient(S.vw / 2, S.vh / 2, Math.min(S.vw, S.vh) * 0.35, S.vw / 2, S.vh / 2, Math.max(S.vw, S.vh) * 0.75);
  g.addColorStop(0, "rgba(0,0,0,0)");
  g.addColorStop(1, "rgba(5,8,12,0.35)");
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, S.vw, S.vh);
  return missing;
}

// 给对局图层的视图信息：缩放、视口、东西环绕的副本范围、世界坐标 → 屏幕坐标
function overlayView(vis) {
  const c = S.cam;
  return {
    z: c.z,
    mode: S.mode, // 当前地图模式（terrain / temperature / …），对局图层据此决定要不要压暗地形
    vw: S.vw,
    vh: S.vh,
    W: S.W,
    H: S.H,
    PW: S.PW,
    k0: Math.floor(vis.x0 / S.PW) - 1,
    k1: Math.floor(vis.x1 / S.PW) + 1,
    x0: vis.x0, x1: vis.x1, y0: vis.y0, y1: vis.y1,
    // 世界坐标（格子半径 = 1，第 k 个东西副本）→ 屏幕坐标
    sx: (wx, k = 0) => (wx + k * S.PW - c.x) * c.z + S.vw / 2,
    sy: (wy) => (wy - c.y) * c.z + S.vh / 2,
    // 某格的地形信息（分块还没加载时为 null）：地形角色、临水方向位掩码、资源 ID
    tile: terrainInfo,
  };
}

// 详情卡和地图使用同一份地形信息，让港口方向、种植园资源的预览保持一致。
export function terrainInfo(col, row) {
  const td = tileData(col, row);
  if (!td) return null;
  const ti = td.ch.terrain[td.li], r = td.ch.resource[td.li];
  return { area: td.ch.area[td.li], role: S.roleOf[ti] ?? null, color: S.meta.ruleset.terrain[ti].color, water: isWater(ti), coast: td.ch.coast[td.li], resource: r ? S.meta.ruleset.resources[r - 1].id : null };
}

// 每帧先按时间预算烤一批位图；有交互（dirty）立刻重画，只是加载有进展（progress）就限频重画。
// 整屏重画（几百张位图 + 河流 + 颗粒暗角）比烤一块位图贵得多：以前每烤 6 块就整屏重画一次，
// 精细地球 375 块要重画六十多次，慢的机器上一次就要几百毫秒，分块请求的回调也被挤得排不上
function frame(now = performance.now()) {
  if (S.renderAt) {
    S.renderCost = now - S.renderAt; // 上次重画那一帧到这一帧的间隔，含光栅化 / 合成
    S.renderAt = 0;
  }
  if (S.overlay?.tick?.(now)) S.dirty = true;
  const t0 = performance.now();
  while (S.bakeQueue.length) {
    const { ch, tier } = S.bakeQueue.shift();
    if (S.chunks.get(chunkKey(ch.cx, ch.cy)) !== ch || getBitmap(ch, tier) || !neighborsReady(ch)) continue;
    bakeBitmap(ch, tier);
    S.progress = true;
    if (performance.now() - t0 > BAKE_MS) break;
  }
  if (S.dirty || (S.progress && now - S.lastRender >= Math.max(LOAD_REDRAW_MS, 2 * S.renderCost))) {
    S.dirty = S.progress = false;
    S.lastRender = S.renderAt = now;
    render();
  }
  requestAnimationFrame(frame);
}

// ---------------------------------------------------------------- 交互

function updateHover(sx, sy) {
  if (!S.meta) return;
  const [wx, wy] = screenToWorld(sx, sy);
  const [col, row] = worldToTile(wx, wy);
  const td = tileData(col, row);
  if (!td) {
    if (S.hover) {
      S.hover = null;
      S.areaEdges = null;
      S.dirty = true;
    }
    S.onHover?.(null, sx, sy);
    return;
  }
  const area = td.ch.area[td.li];
  const info = {
    col: td.col,
    row,
    area,
    terrain: S.meta.ruleset.terrain[td.ch.terrain[td.li]],
    resource: td.ch.resource[td.li] ? S.meta.ruleset.resources[td.ch.resource[td.li] - 1] : null,
    isSea: area >= S.meta.land_areas,
    height: td.ch.height[td.li],
    plate: td.ch.plate[td.li],
    boundary: td.ch.boundary[td.li],
    temp: td.ch.temp[td.li],
    rain: td.ch.rain[td.li],
    wind: windOf(td),
    current: currentOf(td),
    areaNo: area >= S.meta.land_areas ? area - S.meta.land_areas + 1 : area + 1,
  };
  if (!S.hover || S.hover.area !== area) {
    computeAreaEdges(area);
    S.dirty = true;
  }
  S.hover = info;
  S.onHover?.(info, sx, sy);
}

export function setOverlay(overlay) {
  S.overlay = overlay;
  S.dirty = true;
}

export function redraw() {
  S.dirty = true;
}

// Measure actual Canvas drawing at four scales, returning to the player's view.
export async function measureScenePerformance(focus = {}) {
  const saved = {...S.cam};
  const results = [];
  const pause = ms => new Promise(resolve=>setTimeout(resolve,ms));
  try {
    for (const [name,z] of [['世界全景',0],['区域视角',8],['战术视角',32],['建筑近景',80]]) {
      if(!z) fitView();
      else {
        S.cam.z=z;
        centerOn(focus.col??Math.floor(S.W/2),focus.row??Math.floor(S.H/2));
      }
      S.dirty=true;
      const deadline=performance.now()+10000;
      // Load visible terrain and bake tile art before timing steady-state drawing.
      do { S.dirty=true;await pause(60); } while((S.pending.size||S.bakeQueue.length)&&performance.now()<deadline);
      const times=[];
      for(let i=0;i<16;i++) {
        await new Promise(requestAnimationFrame);
        const start=performance.now();render();
        if(i>=4)times.push(performance.now()-start);
      }
      times.sort((a,b)=>a-b);
      results.push({name,zoom:Math.round(S.cam.z*100)/100,samples:times.length,
        median_ms:+times[Math.floor(times.length/2)].toFixed(1),
        p95_ms:+times[Math.ceil(times.length*.95)-1].toFixed(1)});
    }
  } finally {Object.assign(S.cam,saved);S.dirty=true;}
  return results;
}

// 屏幕坐标 → 格子 {col, row}（列已按东西环绕取模）；不在地图上返回 null
export function screenToTile(sx, sy) {
  if (!S.meta) return null;
  const [wx, wy] = screenToWorld(sx, sy);
  const [col, row] = worldToTile(wx, wy);
  if (row < 0 || row >= S.H) return null;
  return { col: ((col % S.W) + S.W) % S.W, row };
}

// 把镜头移到某格（不缩小；比 minZoom 远时拉近到 minZoom）
export function centerOn(col, row, minZoom = 0) {
  if (!S.meta) return;
  S.cam.x = SQRT3 * (col + 0.5 * (row & 1));
  S.cam.y = 1.5 * row;
  if (S.cam.z < minZoom) S.cam.z = minZoom;
  clampCam();
  S.dirty = true;
}

export function setMode(mode) {
  S.mode = mode;
  S.bitmaps.clear();
  S.bitmapPixels = 0;
  S.scaledBitmaps.clear();S.scaledPixels=0;
  S.dirty = true;
}

export function setLayer(name, on) {
  S.layers[name] = on;
  S.bitmaps.clear();
  S.bitmapPixels = 0;
  S.scaledBitmaps.clear();S.scaledPixels=0;
  S.dirty = true;
}

export function init(canvas, { onHover, onClick } = {}) {
  S.canvas = canvas;
  S.ctx = canvas.getContext("2d");
  S.onHover = onHover;
  S.onClick = onClick;
  const resize = () => {
    S.dpr = Math.min(window.devicePixelRatio || 1, 2);
    S.vw = canvas.clientWidth;
    S.vh = canvas.clientHeight;
    canvas.width = Math.round(S.vw * S.dpr);
    canvas.height = Math.round(S.vh * S.dpr);
    if (S.meta) clampCam();
    S.dirty = true;
  };
  new ResizeObserver(resize).observe(canvas);
  resize();

  const pointers = new Map();
  let last = null;
  let pinch = null;
  let press = null; // 按下时的位置和按键：没怎么拖动就松开 = 点击
  canvas.addEventListener("pointerdown", (e) => {
    canvas.setPointerCapture(e.pointerId);
    pointers.set(e.pointerId, [e.offsetX, e.offsetY]);
    last = [e.offsetX, e.offsetY];
    pinch = null;
    press = pointers.size === 1 ? { x: e.offsetX, y: e.offsetY, button: e.button, moved: 0 } : null;
  });
  canvas.addEventListener("pointermove", (e) => {
    if (!pointers.has(e.pointerId)) {
      updateHover(e.offsetX, e.offsetY);
      return;
    }
    pointers.set(e.pointerId, [e.offsetX, e.offsetY]);
    if (pointers.size === 2) {
      const [a, b] = [...pointers.values()];
      const dist = Math.hypot(a[0] - b[0], a[1] - b[1]);
      const mid = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];
      if (pinch) {
        zoomBy(dist / pinch.dist, mid[0], mid[1]);
        panBy(mid[0] - pinch.mid[0], mid[1] - pinch.mid[1]);
      }
      pinch = { dist, mid };
      return;
    }
    if (press) press.moved = Math.max(press.moved, Math.hypot(e.offsetX - press.x, e.offsetY - press.y));
    if (press && press.moved < 6) return; // 还没确定是拖动，先不平移
    if (last) panBy(e.offsetX - last[0], e.offsetY - last[1]);
    last = [e.offsetX, e.offsetY];
    canvas.classList.add("dragging");
  });
  const up = (e) => {
    if (e.type === "pointerup" && press && press.moved < 6 && pointers.size === 1) {
      const tile = screenToTile(e.offsetX, e.offsetY);
      S.onClick?.(tile, press.button, e.offsetX, e.offsetY);
    }
    press = null;
    pointers.delete(e.pointerId);
    if (pointers.size === 0) {
      last = null;
      canvas.classList.remove("dragging");
    }
    pinch = null;
  };
  canvas.addEventListener("pointerup", up);
  canvas.addEventListener("pointercancel", up);
  canvas.addEventListener("pointerleave", () => S.onHover?.(null));
  canvas.addEventListener(
    "wheel",
    (e) => {
      e.preventDefault();
      zoomBy(Math.exp(-e.deltaY * (e.deltaMode ? 0.05 : 0.0018)), e.offsetX, e.offsetY);
    },
    { passive: false },
  );
  canvas.addEventListener("dblclick", (e) => zoomBy(2, e.offsetX, e.offsetY));
  window.addEventListener("keydown", (e) => {
    if (e.target.closest("input, select, textarea")) return;
    const step = 80;
    if (e.key === "+" || e.key === "=") zoomBy(1.25);
    else if (e.key === "-") zoomBy(0.8);
    else if (e.key === "ArrowLeft") panBy(step, 0);
    else if (e.key === "ArrowRight") panBy(-step, 0);
    else if (e.key === "ArrowUp") panBy(0, step);
    else if (e.key === "ArrowDown") panBy(0, -step);
  });
  requestAnimationFrame(frame);
}

export const _debug = S;
