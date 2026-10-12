"""生成架空世界的名字库 rules/alt_ww2/names.yaml（东方 / 西方两个系列）。

    uv run python tools/names/generate.py

每个系列：20 个国名、1000 个城市名、1000 个人名（将领）。城市名由“前字 + 后缀”、西式音节
拼出来，按固定种子抽样，所以每次生成结果一样；手工挑过的字表在下面，改字表重新生成即可。
国名和城市名在同一个系列里可以互用（城名库用完借国名、独立建国借城名），见 engine/systems/names.py。
"""

from __future__ import annotations

import random
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "rules" / "alt_ww2" / "names.yaml"
COUNT = 1000

EAST_NATIONS = [
    "大夏帝国", "南楚王国", "西凉国", "北燕王朝", "高昌国", "苍梧国", "青丘国", "昆仑帝国",
    "蓬莱国", "华胥国", "九黎联邦", "瀛洲国", "玄武王朝", "朱雀国", "云梦国", "天穹帝国",
    "龙城国", "白虎王朝", "岭南联邦", "凤翔共和国",
]  # fmt: skip
WEST_NATIONS = [
    "洛塔林吉亚公国", "卡兰西亚帝国", "赫斯佩里亚联邦", "诺瓦里斯王国", "塞伦迪亚共和国",
    "维斯特法伦邦联", "阿奎塔尼亚王国", "卢米纳联合王国", "奥斯特马克公国", "格拉纳迪亚王国",
    "斯卡迪亚联盟", "提罗利亚自由邦", "伊比利恩王国", "哈布斯塔德帝国", "卡林西亚公国",
    "莫拉维亚共和国", "弗里斯兰联邦", "勃兰登堡选侯国", "伊斯特里亚共和国", "萨伏伊亚王国",
]  # fmt: skip

# ---- 东方城市：前字（山川、方位、吉祥字）+ 后缀（州、阳、关、津……），另有少量三字名
EAST_HEAD = (
    "青云江河山海东西南北安宁平永长新临清白金玉龙凤鹤松柏桂兰宜华瑞庆昌盛定靖怀武文德建兴嘉泰和丰"
    "汉秦楚齐燕赵越吴湘漠沙雁鹰霞雪霜石岩泉溪湖林原峰岭川苍翠碧丹紫赤黛乌银铜铁宝珠明光昭晋"
    "淮洛渭汾泾沅澧漳湄滨湛潮渝岷峨衡嵩岱恒崇巍荆襄郢鄂夔蓟冀并凉肃甘瓜沙伊"
)
EAST_TAIL = (
    "州阳安宁城江川山原关门口津港浦湾陵都京平丰昌泉溪林岭峰谷台堡屯镇渡滩洲塘桥河湖源野坪岗寨亭庄"
)
EAST_TRIPLE_HEAD = [
    "长",
    "新",
    "大",
    "小",
    "东",
    "西",
    "南",
    "北",
    "上",
    "下",
    "古",
    "望",
    "临",
    "白",
    "黑",
]
EAST_TRIPLE_MID = "云河山海江石沙水阳林泉风雪月星龙凤马鹿"
EAST_TRIPLE_TAIL = "口关城镇渡湾堡寨坪岭"

# ---- 西方城市：音译音节拼成，偶尔加“圣 / 新”前缀或“堡 / 港 / 维尔……”后缀
WEST_START = [
    "阿", "巴", "贝", "博", "布", "卡", "科", "达", "德", "杜", "埃", "菲", "弗", "加", "格", "哈",
    "赫", "伊", "克", "拉", "莱", "罗", "卢", "马", "梅", "莫", "纳", "尼", "诺", "奥", "帕", "佩",
    "波", "里", "萨", "塞", "索", "塔", "特", "托", "维", "沃", "泽", "艾", "欧", "温", "汉", "霍",
]  # fmt: skip
WEST_MID = [
    "尔", "兰", "林", "伦", "斯", "瑞", "里", "拉", "勒", "纳", "诺", "维", "塔", "卡", "德", "多",
    "米", "蒙", "贝", "罗", "西", "瓦", "利", "莱", "萨", "托", "门", "坎", "桑", "姆", "丁", "格",
]  # fmt: skip
WEST_END = [
    "堡", "港", "维尔", "顿", "斯特", "菲尔德", "斯塔德", "霍姆", "加德", "亚", "那", "诺", "拉", "斯",
    "尔", "恩", "森", "曼", "伯格", "海姆", "瓦", "利亚", "蒂亚", "茨", "克", "多", "里", "桥", "湾",
]  # fmt: skip

# ---- 人名
EAST_SURNAMES = (
    "王李张刘陈杨赵黄周吴徐孙马胡朱郭何罗高林郑梁谢宋唐许韩冯邓曹彭曾萧田董袁潘蒋蔡余杜叶程苏魏吕丁"
    "任沈姚卢姜崔钟谭陆汪范金石廖贾夏韦付方白邹孟熊秦邱江尹薛段雷侯龙史陶黎贺顾毛郝龚邵万钱严覃武戴"
    "莫孔向汤岳霍卫季聂柳邢欧施"
)
EAST_COMPOUND = [
    "欧阳",
    "司马",
    "诸葛",
    "上官",
    "慕容",
    "令狐",
    "皇甫",
    "宇文",
    "长孙",
    "尉迟",
    "公孙",
    "东方",
]
EAST_GIVEN = (
    "振国德明志远云鹏启安承武威忠毅雄光建中山岳飞廷宗瑞伯仲昭烈霆钧铮澜峰嵩松柏宇骁锋骏桓靖宏烨"
    "文博翰林泽涛海川晨曦星辰怀瑾瑜璋琦珩衡钰铭昊旭晟睿哲敬诚恒谦乾坤勇刚强伟彬斌嘉祥祺"
)
WEST_GIVEN = [
    "亚瑟", "埃德蒙", "卡尔", "弗里德里希", "亨利", "路易", "奥托", "威廉", "乔治", "查理", "爱德华",
    "马克西姆", "伊万", "米哈伊尔", "安东", "阿列克谢", "维克多", "让", "皮埃尔", "安德烈", "费利克斯",
    "古斯塔夫", "汉斯", "约瑟夫", "利奥波德", "马蒂亚斯", "康拉德", "西奥多", "弗朗茨", "雨果", "朱利安",
    "拉斐尔", "卢卡斯", "马可", "安东尼奥", "费尔南多", "迭戈", "卡洛斯", "塞巴斯蒂安", "罗德里戈",
    "埃里克", "斯文", "奥拉夫", "比约恩", "拉尔斯", "托马斯", "彼得", "保罗", "雅各布", "西蒙", "丹尼尔",
    "塞缪尔", "本杰明", "奥古斯特", "克劳德", "加斯东", "热拉尔", "蒙哥马利", "巴兹尔", "温斯顿",
]  # fmt: skip
WEST_SURNAMES = [
    "施密特", "霍夫曼", "布朗", "格林", "莫罗", "杜邦", "罗西", "巴尔加斯", "卡斯特罗", "伊万诺夫",
    "彼得罗夫", "安德森", "汉森", "冯·施泰因", "冯·克莱斯特", "德·拉图尔", "费舍尔", "韦伯", "瓦格纳",
    "贝克尔", "舒尔茨", "科赫", "里希特", "克莱因", "沃尔夫", "勒克莱尔", "贝尔纳", "马丁", "杜兰",
    "加西亚", "费尔南德斯", "洛佩斯", "桑切斯", "里韦拉", "孔蒂", "马里尼", "费拉里", "布鲁诺", "加洛",
    "诺瓦克", "科瓦尔斯基", "维什涅夫斯基", "索科洛夫", "库兹涅佐夫", "波波夫", "莫罗佐夫", "沃尔科夫",
    "尼尔森", "约翰松", "林德伯格", "斯特兰德", "霍尔姆", "奥康纳", "麦克唐纳", "坎贝尔", "斯图尔特",
    "汉密尔顿", "布莱克", "怀特", "惠特曼", "哈里森", "埃利奥特", "蒙特罗斯", "范德堡", "范·戴克",
    "德弗里斯", "扬森", "奥尔森", "卡尔松", "贝格曼", "席勒", "布劳恩", "海涅", "隆美尔", "曼施坦因",
]  # fmt: skip


def _pick(rng, pool, n, taken):
    pool = [x for x in dict.fromkeys(pool) if x not in taken]
    rng.shuffle(pool)
    if len(pool) < n:
        raise SystemExit(f"候选不够：要 {n}，只有 {len(pool)}")
    out = pool[:n]
    taken.update(out)
    return out


def east_cities(rng, taken):
    two = [h + t for h in EAST_HEAD for t in EAST_TAIL if h != t]
    three = [a + b + c for a in EAST_TRIPLE_HEAD for b in EAST_TRIPLE_MID for c in EAST_TRIPLE_TAIL]
    names = _pick(rng, two, COUNT - 150, taken) + _pick(rng, three, 150, taken)
    rng.shuffle(names)
    return names


def west_cities(rng, taken):
    pool = []
    for a in WEST_START:
        for b in WEST_MID:
            for c in WEST_END:
                name = a + b + c
                if 2 <= len(name) <= 6 and a != b:
                    pool.append(name)
            pool.append(a + b + "斯")
    singles = _pick(rng, pool, COUNT - 120, taken)
    saints = _pick(rng, ["圣" + x for x in pool if len(x) <= 4], 60, taken)
    news = _pick(rng, ["新" + x for x in pool if len(x) <= 4], 60, taken)
    names = singles + saints + news
    rng.shuffle(names)
    return names


def east_people(rng):
    pool = [s + g for s in EAST_SURNAMES for g in EAST_GIVEN]
    pool += [s + a + b for s in EAST_SURNAMES for a in EAST_GIVEN[:40] for b in EAST_GIVEN[40:80]]
    pool += [s + g for s in EAST_COMPOUND for g in EAST_GIVEN]
    return _pick(rng, pool, COUNT, set())


def west_people(rng):
    return _pick(rng, [f"{g}·{s}" for g in WEST_GIVEN for s in WEST_SURNAMES], COUNT, set())


# ---- 舰名（战舰造出来时从本系列里挑，engine/systems/names.py 的 ship_name）
SHIP_COUNT = 200
EAST_SHIP_HEAD = "定镇致经靖济平威扬超宁肇永逸应建飞振广保武安海长鸿腾凌御"
EAST_SHIP_TAIL = "远海安宁威勇捷胜清和平霆云龙鹏涛澜疆"
WEST_SHIPS = [
    "无畏", "不屈", "胜利", "光辉", "可畏", "坚定", "勇敢", "凯旋", "荣耀", "决心", "复仇", "反击",
    "狂怒", "无敌", "不倦", "警惕", "果敢", "雷霆", "风暴", "飓风", "闪电", "利剑", "长矛", "猎户座",
    "天狼星", "北极星", "阿波罗", "海神", "雅典娜", "密涅瓦", "朱诺", "狄安娜", "玛尔斯", "赫拉克勒斯",
    "阿喀琉斯", "阿贾克斯", "尤利西斯", "埃涅阿斯", "无惧", "坚毅", "英勇", "荣誉", "正义", "自由",
    "团结", "黎明", "晨星", "暮光", "海燕", "信天翁", "雄鹰", "猎鹰", "渡鸦", "狮心", "王冠", "权杖",
    "宝剑", "盾牌", "铁甲", "征服者", "守护者", "开拓者", "远征", "冒险", "探索", "发现", "进取",
    "挑战", "胜利女神", "凤凰", "独角兽", "狮鹫", "巨人", "泰坦", "海妖", "三叉戟", "堡垒", "哨兵",
]  # fmt: skip


def east_ships(rng):
    pool = [h + t for h in EAST_SHIP_HEAD for t in EAST_SHIP_TAIL if h != t]
    return [n + "号" for n in _pick(rng, pool, SHIP_COUNT, set())]


def west_ships(rng):
    # 美德、星辰、神话的名字，再加上本系列的名和姓（“威廉号”“施密特号”）
    people = [s.split("·")[-1] for s in WEST_SURNAMES] + list(WEST_GIVEN)
    pool = list(
        dict.fromkeys(WEST_SHIPS + [p for p in people if len(p) >= 2])
    )  # “让号”这种单字名不用
    return [n + "号" for n in _pick(rng, pool, SHIP_COUNT, set())]


def main():
    rng = random.Random(20261010)
    taken: set[str] = set(EAST_NATIONS) | set(WEST_NATIONS)
    # 不和国家模板里已有的城市名、国名重复
    templates = yaml.safe_load((ROOT / "rules" / "alt_ww2" / "nations.yaml").read_text("utf-8"))
    for t in templates.values():
        taken.add(t["name"])
        taken.update(t["cities"])
    data = {
        "east": {
            "nations": EAST_NATIONS,
            "cities": east_cities(rng, taken),
            "people": east_people(rng),
        },
        "west": {
            "nations": WEST_NATIONS,
            "cities": west_cities(rng, taken),
            "people": west_people(rng),
        },
    }
    # 舰名放在最后抽：不改变前面国名、城市名、人名的抽样结果
    data["east"]["ships"] = east_ships(rng)
    data["west"]["ships"] = west_ships(rng)
    head = (
        "# 名字库（架空）：东方 / 西方两个系列，各 20 个国名、1000 个城市名、1000 个人名（将领）、"
        "200 个舰名。\n"
        "# 由 tools/names/generate.py 生成；国名和城市名在同一系列里可以互用（见 engine/systems/names.py）。\n"
    )
    body = yaml.safe_dump(data, allow_unicode=True, sort_keys=False, width=200)
    OUT.write_text(head + body, encoding="utf-8")
    for side in ("east", "west"):
        print(
            side,
            {k: len(v) for k, v in data[side].items()},
            data[side]["cities"][:8],
            data[side]["people"][:5],
        )


if __name__ == "__main__":
    main()
