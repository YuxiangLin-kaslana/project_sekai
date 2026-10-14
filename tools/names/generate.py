"""生成架空世界的名字库 rules/alt_ww2/names.yaml（东方 / 西方两个系列）。

    uv run python tools/names/generate.py

每个系列：200 个国名（按文化小类，同时写出 rules/alt_ww2/cultures.yaml）、2000 个城市名、1000 个人名（将领）、
200 个舰名。城市名由“前字 + 后缀”、西式音节拼出来，按固定种子抽样，所以每次生成结果一样；手工挑过的字表
在下面，改字表重新生成即可。
国名和城市名在同一个系列里可以互用（城名库用完借国名、独立建国借城名），见 engine/systems/names.py。
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))  # 用引擎的 base_name（国名去掉政体后缀）
OUT = ROOT / "rules" / "alt_ww2" / "names.yaml"
CULTURE_OUT = ROOT / "rules" / "alt_ww2" / "cultures.yaml"
COUNT = 2000  # 每个系列的城市名
PEOPLE = 1000  # 每个系列的人名（将领）

# ---- 国名：按文化小类（cultures.yaml）生成。每个小类：名额、手写的词干（历史 / 地理风格）、不够时按构词
# 规则补的词干、政体后缀（按权重抽）。原来手写的 40 个国名保留在各小类里。
EAST_NATION_COUNT = WEST_NATION_COUNT = 200
CULTURES = {
    # ---- 东方
    "zhongyuan": ("中原", "east", 45, [
        "大夏帝国", "北燕王朝", "华胥国", "玄武王朝", "天穹帝国", "白虎王朝", "凤翔共和国",
    ], [
        "轩辕", "有熊", "涿鹿", "中山", "常山", "琅琊", "渤海", "陈留", "颍川", "弘农", "太原", "上党",
        "河东", "河内", "京兆", "平原", "济阴", "东郡", "代郡", "雁门", "云中", "北地", "陇西", "天水",
    ], ("大北南东西后前新中", "夏商周秦汉晋隋唐宋齐梁魏燕赵韩鲁卫郑陈蔡曹辽")),
    "jingchu": ("荆楚", "east", 35, [
        "南楚王国", "苍梧国", "青丘国", "云梦国", "朱雀国", "九黎联邦",
    ], [
        "巴", "蜀", "庸", "随", "夔", "郢", "东瓯", "闽越", "南越", "句吴", "于越", "夜郎", "且兰", "句町",
        "牂牁", "百越", "荆南", "武陵", "零陵", "桂阳", "豫章", "庐陵", "江夏", "洞庭", "鄱阳", "湘楚",
    ], ("南东西云江湘大新", "楚吴越巴蜀湘闽滇黔粤瓯")),
    "xiyu": ("西域", "east", 35, [
        "西凉国", "高昌国", "昆仑帝国", "龙城国",
    ], [
        "龟兹", "于阗", "疏勒", "楼兰", "焉耆", "车师", "月氏", "乌孙", "大宛", "康居", "莎车", "精绝",
        "鄯善", "且末", "温宿", "姑墨", "柔然", "突厥", "回纥", "鲜卑", "契丹", "吐谷浑", "党项", "敕勒",
        "坚昆", "黠戛斯", "粟特", "葱岭", "祁连", "敦煌", "玉门", "河西", "伊吾",
    ], ("北西东大", "凉戎狄羌胡")),
    "dongying": ("东瀛", "east", 25, [
        "蓬莱国", "瀛洲国",
    ], [
        "扶桑", "出云", "筑紫", "虾夷", "琉球", "越后", "甲斐", "肥前", "萨摩", "长门", "陆奥", "常陆",
        "武藏", "相模", "伊势", "吉备", "丰后", "日向", "土佐", "伊予", "尾张", "美浓", "信浓", "若狭",
        "加贺", "但马", "播磨", "纪伊", "隐岐", "对马",
    ], ("东北南西", "瀛津岛扶")),
    "tianzhu": ("天竺", "east", 30, [], [
        "摩揭陀", "拘萨罗", "犍陀罗", "跋耆", "迦湿弥罗", "羯陵伽", "案达罗", "朱罗", "潘地亚", "遮娄其",
        "笈多", "贵霜", "迦尸", "般遮罗", "俱卢", "阿槃提", "苏剌侘", "摩腊婆", "马拉塔", "曲女",
        "憍赏弥", "吠舍离", "室罗伐", "信德", "旁遮普", "古吉拉特", "孟加拉", "奥里萨", "喀拉拉", "迈索尔",
        "维杰耶", "那烂陀",
    ], ("大南北东西", "竺陀罗伽")),
    "nanyang": ("南洋", "east", 30, [
        "岭南联邦",
    ], [
        "真腊", "占婆", "扶南", "室利佛逝", "满者伯夷", "蒲甘", "骠", "苏禄", "吕宋", "马打蓝", "勃泥",
        "柔佛", "马六甲", "亚齐", "爪哇", "巴厘", "三佛齐", "单马令", "盘盘", "丹丹", "罗斛", "哀牢",
        "南掌", "交趾", "九真", "日南", "林邑", "澜沧", "兰纳", "婆罗", "苏门",
    ], ("南东西大新", "洋岛椰屿")),
    # ---- 西方
    "latin": ("拉丁", "west", 40, [
        "卡兰西亚帝国", "赫斯佩里亚联邦", "诺瓦里斯王国", "阿奎塔尼亚王国", "卢米纳联合王国",
        "格拉纳迪亚王国", "伊比利恩王国", "萨伏伊亚王国", "伊斯特里亚共和国",
    ], [
        "伦巴第亚", "托斯卡尼亚", "翁布里亚", "艾米利亚", "利古里亚", "皮埃蒙特", "威尼提亚", "卡拉布里亚",
        "阿普利亚", "坎帕尼亚", "西西里亚", "撒丁尼亚", "科西加", "普罗旺斯", "朗格多克", "加斯科涅",
        "布列塔尼亚", "诺曼底亚", "阿拉贡", "纳瓦拉", "莱昂", "加利西亚", "安达卢西亚", "穆尔西亚",
        "瓦伦西亚", "阿斯图里亚", "坎塔布里亚", "阿尔加维", "达尔马提亚", "潘诺尼亚", "达基亚",
        "瓦拉几亚", "摩尔达维亚", "特兰西瓦尼亚",
    ], (["维", "卡", "塞", "马", "阿", "萨", "托", "莱", "奥", "普", "贝", "科"],
        ["罗", "斯", "拉", "里", "塔", "尼", "西", "蒙", "瓦", "达"],
        ["利亚", "尼亚", "塔尼亚", "西亚", "迪亚", "提亚"])),
    "germanic": ("日耳曼", "west", 40, [
        "洛塔林吉亚公国", "维斯特法伦邦联", "奥斯特马克公国", "提罗利亚自由邦", "哈布斯塔德帝国",
        "弗里斯兰联邦", "勃兰登堡选侯国", "卡林西亚公国",
    ], [
        "巴伐利亚", "萨克森", "图林根", "黑森", "符腾堡", "波美拉尼亚", "梅克伦堡", "施瓦本", "法兰克尼亚",
        "莱茵兰", "汉诺威", "不伦瑞克", "奥尔登堡", "荷尔斯泰因", "石勒苏益格", "安哈尔特", "利珀",
        "拿骚", "普法尔茨", "巴登", "阿尔萨斯", "卢森堡", "施蒂利亚", "福拉尔贝格", "萨尔茨堡", "帕绍",
        "雷根斯堡", "奥格斯堡", "美因茨", "特里尔", "明斯特", "不来梅", "吕贝克",
    ], (["诺德", "祖德", "维斯特", "奥斯特", "霍恩", "林登", "罗森", "埃伦", "瓦尔德", "施坦"],
        ["马克", "兰", "堡", "海姆", "瑙", "费尔斯"])),
    "nordic": ("北欧", "west", 25, [
        "斯卡迪亚联盟",
    ], [
        "斯堪尼亚", "哥特兰", "斯韦阿", "约塔兰", "诺尔兰", "雅姆特兰", "达拉纳", "乌普兰", "特伦德拉格",
        "霍达兰", "罗加兰", "阿格德尔", "芬马克", "赫尔辛兰", "卡累利阿", "萨沃", "塔瓦斯提亚", "拉普兰",
        "法罗", "设得兰", "奥克尼", "赫布里底", "日德兰", "菲英", "西兰", "博恩霍尔姆",
    ], (["斯瓦", "冰", "雾", "霜", "峡", "诺尔", "雅尔", "索尔"], ["加德", "维克", "海姆", "兰", "峡湾"])),
    "anglo": ("盎格鲁", "west", 30, [
        "塞伦迪亚共和国",
    ], [
        "威塞克斯", "麦西亚", "诺森布里亚", "东盎格利亚", "肯特", "苏塞克斯", "埃塞克斯", "康沃尔",
        "坎布里亚", "斯特拉斯克莱德", "达尔里亚达", "加勒多尼亚", "格温内思", "波伊斯", "德赫巴斯",
        "林迪西", "赫威赛", "德拉", "伯尼西亚", "埃尔梅特", "雷格德", "多姆诺尼亚", "迪弗德", "布雷克尼",
        "新威塞克斯", "新加勒多尼亚", "维多利亚",
    ], (["新", "北", "南", "东", "西"], ["阿尔比恩", "威塞克斯", "麦西亚", "肯特", "康沃尔", "约克"])),
    "slavic": ("斯拉夫", "west", 35, [
        "莫拉维亚共和国",
    ], [
        "波希米亚", "西里西亚", "哈利奇", "沃里尼亚", "诺夫哥罗德", "普斯科夫", "弗拉基米尔", "梁赞",
        "特维尔", "斯摩棱斯克", "切尔尼戈夫", "波洛茨克", "鲁塞尼亚", "卢萨蒂亚", "斯拉沃尼亚", "拉什卡",
        "泽塔", "杜克利亚", "马佐维亚", "库雅维亚", "波德拉谢", "顿河", "扎波罗热", "库班", "第聂伯",
        "伏尔加", "卡尔巴阡", "维斯瓦", "奥得", "斯拉维亚", "波多利亚", "白罗斯", "红罗斯", "斯洛文",
        "卢布林", "维捷布斯克", "图罗夫", "克罗巴提亚",
    ], (["斯维亚托", "博列", "拉多", "雅罗", "斯坦尼"], ["斯拉维亚", "格勒", "斯拉夫"])),
    "levant": ("近东", "west", 30, [], [
        "亚述", "巴比伦", "腓尼基", "迦南", "吕底亚", "弗里吉亚", "卡帕多奇亚", "本都", "奇里乞亚",
        "示巴", "阿克苏姆", "米底", "埃兰", "乌拉尔图", "科尔基斯", "帕提亚", "巴克特里亚", "帕尔米拉",
        "纳巴泰", "希贾兹", "内志", "阿曼", "马格里布", "毛里塔尼亚", "努米底亚", "昔兰尼加", "的黎波里",
        "美索", "苏萨", "埃德萨", "安条克",
    ], (["阿尔", "巴", "卡", "萨", "哈", "法", "马", "达"], ["斯坦", "巴德", "利亚", "西亚", "希尔"])),
}  # fmt: skip
# 政体后缀（权重）：东方、西方各一套；近东另有苏丹、埃米尔
SUFFIXES = {
    "zhongyuan": [
        ("国", 30),
        ("王朝", 20),
        ("帝国", 20),
        ("王国", 15),
        ("共和国", 10),
        ("联邦", 5),
    ],
    "jingchu": [("国", 40), ("王国", 25), ("联邦", 15), ("共和国", 15), ("帝国", 5)],
    "xiyu": [("国", 45), ("汗国", 30), ("王国", 15), ("联盟", 10)],
    "dongying": [("国", 45), ("皇国", 20), ("藩国", 20), ("王国", 15)],
    "tianzhu": [("王国", 35), ("帝国", 20), ("联邦", 20), ("邦联", 15), ("共和国", 10)],
    "nanyang": [("国", 40), ("王国", 30), ("联邦", 20), ("苏丹国", 10)],
    "latin": [("王国", 35), ("公国", 20), ("共和国", 20), ("联邦", 15), ("帝国", 10)],
    "germanic": [
        ("公国", 30),
        ("选侯国", 15),
        ("邦联", 15),
        ("帝国", 10),
        ("自由邦", 15),
        ("王国", 15),
    ],
    "nordic": [("王国", 40), ("联盟", 20), ("议会国", 15), ("联邦", 25)],
    "anglo": [("王国", 35), ("联合王国", 15), ("合众国", 15), ("联邦", 20), ("自治领", 15)],
    "slavic": [
        ("公国", 30),
        ("人民国", 15),
        ("共和国", 20),
        ("联盟", 15),
        ("大公国", 10),
        ("沙皇国", 10),
    ],
    "levant": [("苏丹国", 30), ("王国", 25), ("帝国", 20), ("埃米尔国", 15), ("哈里发国", 10)],
}


def _stem_ok(stem, taken_stems):
    return 1 <= len(stem) <= 6 and stem not in taken_stems


def culture_nations(rng, taken):
    """每个文化小类的国名：原有的手写国名 + 手写词干配政体后缀，不够再按构词规则补；全局不重名（词干也不重）。
    返回 {小类: [国名…]}。"""
    from engine.systems.names import base_name

    out = {}
    stems_taken = {base_name(n) for n in taken}
    for cid, (_, _, quota, keep, stems, parts) in CULTURES.items():
        names = [n for n in keep if n not in taken]
        stems_taken |= {base_name(n) for n in names}
        weights = SUFFIXES[cid]
        generated = [
            "".join(combo)
            for combo in (
                [(a, b) for a in parts[0] for b in parts[1]]
                if len(parts) == 2
                else [(a, b, c) for a in parts[0] for b in parts[1] for c in parts[2]]
            )
            if combo[0] != combo[1]
        ]
        rng.shuffle(generated)
        for stem in [*stems, *generated]:
            if len(names) >= quota:
                break
            if not _stem_ok(stem, stems_taken):
                continue
            suffix = rng.choices([s for s, _ in weights], [w for _, w in weights])[0]
            name = stem + suffix
            if name in taken or base_name(name) not in (stem, name):
                continue  # 去政体后缀以后不是原词干（会和别的国名撞）：跳过
            names.append(name)
            stems_taken |= {stem, base_name(name)}
        if len(names) < quota:
            raise SystemExit(f"{cid}：国名不够，要 {quota}，只有 {len(names)}")
        taken.update(names)
        out[cid] = names
    return out


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
    names = _pick(rng, two, COUNT - 300, taken) + _pick(rng, three, 300, taken)
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
    singles = _pick(rng, pool, COUNT - 240, taken)
    saints = _pick(rng, ["圣" + x for x in pool if len(x) <= 4], 120, taken)
    news = _pick(rng, ["新" + x for x in pool if len(x) <= 4], 120, taken)
    names = singles + saints + news
    rng.shuffle(names)
    return names


def east_people(rng):
    pool = [s + g for s in EAST_SURNAMES for g in EAST_GIVEN]
    pool += [s + a + b for s in EAST_SURNAMES for a in EAST_GIVEN[:40] for b in EAST_GIVEN[40:80]]
    pool += [s + g for s in EAST_COMPOUND for g in EAST_GIVEN]
    return _pick(rng, pool, PEOPLE, set())


def west_people(rng):
    return _pick(rng, [f"{g}·{s}" for g in WEST_GIVEN for s in WEST_SURNAMES], PEOPLE, set())


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
    from engine.systems.names import base_name

    rng = random.Random(20261010)
    taken: set[str] = set()
    # 不和国家模板里已有的城市名、国名（连同去掉政体的词干）重复
    templates = yaml.safe_load((ROOT / "rules" / "alt_ww2" / "nations.yaml").read_text("utf-8"))
    for t in templates.values():
        taken |= {t["name"], base_name(t["name"])}
        taken.update(t["cities"])
    by_culture = culture_nations(rng, taken)
    taken |= {
        base_name(n) for names in by_culture.values() for n in names
    }  # 城市名也不和国名词干撞
    family = {cid: spec[1] for cid, spec in CULTURES.items()}
    data = {
        side: {
            "nations": [
                n for cid, names in by_culture.items() if family[cid] == side for n in names
            ]
        }
        for side in ("east", "west")
    }
    data["east"]["cities"] = east_cities(rng, taken)
    data["east"]["people"] = east_people(rng)
    data["west"]["cities"] = west_cities(rng, taken)
    data["west"]["people"] = west_people(rng)
    # 舰名放在最后抽：不改变前面国名、城市名、人名的抽样结果
    data["east"]["ships"] = east_ships(rng)
    data["west"]["ships"] = west_ships(rng)
    head = (
        f"# 名字库（架空）：东方 / 西方两个系列，各 {EAST_NATION_COUNT} 个国名（按文化小类，见 cultures.yaml）、"
        f"{COUNT} 个城市名、{PEOPLE} 个人名（将领）、{SHIP_COUNT} 个舰名。\n"
        "# 由 tools/names/generate.py 生成；国名和城市名在同一系列里可以互用（见 engine/systems/names.py）。\n"
    )
    body = yaml.safe_dump(data, allow_unicode=True, sort_keys=False, width=200)
    OUT.write_text(head + body, encoding="utf-8")
    # 文化小类：名字、大类、名字库里属于它的国名（国家模板的小类写在 nations.yaml）
    cultures = {
        cid: {"name": spec[0], "family": spec[1], "nations": by_culture[cid]}
        for cid, spec in CULTURES.items()
    }
    CULTURE_OUT.write_text(
        "# 文化（engine/systems/culture.py）：大类 family 是东方 east / 西方 west（和 names.yaml 的名字系列对应），\n"
        "# 下面分小类。国家模板在 nations.yaml 里写 subculture；名字库里的国名列在这里的 nations 下（由\n"
        "# tools/names/generate.py 生成）；都没列到的国名，按国名在同一大类的小类里确定地分一个。分裂出来的\n"
        "# 新政权继承首都那座城的文化（见 engine/systems/secession.py）。\n"
        "# 外交：同小类的国家关系好，同大类其次，不同大类差——关系目标值上的加减项（见 engine/systems/diplomacy.py）。\n"
        + yaml.safe_dump(cultures, allow_unicode=True, sort_keys=False, width=200),
        encoding="utf-8",
    )
    for side in ("east", "west"):
        print(side, {k: len(v) for k, v in data[side].items()}, data[side]["cities"][:6])
    for cid, names in by_culture.items():
        print(cid, len(names), names[:10])


if __name__ == "__main__":
    main()
