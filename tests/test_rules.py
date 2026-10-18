import shutil
from pathlib import Path

import pytest
import yaml

from engine.rules.loader import RULES_ROOT, RulesetError, load_ruleset


@pytest.fixture
def rules_root(tmp_path: Path) -> Path:
    shutil.copytree(RULES_ROOT / "alt_ww2", tmp_path / "alt_ww2")
    shutil.copytree(RULES_ROOT / "_world", tmp_path / "_world")  # 共用的地形与世界生成
    return tmp_path


def _edit(path: Path, fn) -> None:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    fn(data)
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")


def test_builtin_ruleset_loads():
    rules = load_ruleset("alt_ww2")
    assert rules.id.startswith("alt_ww2@0.1.0#")
    assert rules.terrain_index(rules.worldgen.biomes.ocean) >= 0
    assert rules.resource_at(rules.resource_index("fish")).id == "fish"
    assert rules.resource_at(0) is None


def test_id_hash_tracks_content(rules_root: Path):
    before = load_ruleset("alt_ww2", root=rules_root).id
    _edit(rules_root / "_world/terrain.yaml", lambda d: d["plains"].update(defense=5))
    assert load_ruleset("alt_ww2", root=rules_root).id != before


def test_unknown_terrain_in_resource_is_rejected(rules_root: Path):
    _edit(rules_root / "alt_ww2/resources.yaml", lambda d: d["iron"].update(terrains=["lava"]))
    with pytest.raises(RulesetError, match="lava"):
        load_ruleset("alt_ww2", root=rules_root)


def test_biome_must_match_water_flag(rules_root: Path):
    _edit(
        rules_root / "_world/ruleset.yaml",
        lambda d: d["worldgen"]["biomes"].update(ocean="plains"),
    )
    with pytest.raises(RulesetError, match="ocean"):
        load_ruleset("alt_ww2", root=rules_root)


def test_unknown_field_is_rejected(rules_root: Path):
    _edit(rules_root / "_world/terrain.yaml", lambda d: d["plains"].update(speed=3))
    with pytest.raises(RulesetError):
        load_ruleset("alt_ww2", root=rules_root)


def test_extends_overrides_and_deletes(rules_root: Path):
    child = rules_root / "fantasy"
    child.mkdir()
    (child / "ruleset.yaml").write_text(
        "name: fantasy\nversion: 0.0.1\ntitle: 测试\nextends: alt_ww2\nworldgen:\n  land_ratio: 0.5\n",
        encoding="utf-8",
    )
    (child / "resources.yaml").write_text(
        "oil: null\nmithril:\n  name: 秘银\n  kind: strategic\n  terrains: [mountains]\n"
        "  frequency: 5\n  color: '#aabbcc'\n",
        encoding="utf-8",
    )
    # 删掉石油后，引用它的单位和设施也要删掉（否则引用检查不通过）
    (child / "units.yaml").write_text(
        "armor: null\nanti_tank: null\nbattleship: null\ncarrier: null\n", encoding="utf-8"
    )
    (child / "buildings.yaml").write_text("oil_well: null\n", encoding="utf-8")
    # 给这些单位加成的科技也删掉
    (child / "techs.yaml").write_text(
        "armored_warfare: null\nheavy_armor: null\nbattleships: null\nproximity_fuze: null\n",
        encoding="utf-8",
    )
    rules = load_ruleset("fantasy", root=rules_root)
    assert rules.worldgen.land_ratio == 0.5
    assert (
        rules.worldgen.mountain_ratio
        == load_ruleset("alt_ww2", root=rules_root).worldgen.mountain_ratio
    )
    assert "oil" not in rules.resources
    assert "mithril" in rules.resources
    assert "plains" in rules.terrain


def test_extends_cycle_is_rejected(rules_root: Path):
    for name, parent in (("a", "b"), ("b", "a")):
        d = rules_root / name
        d.mkdir()
        (d / "ruleset.yaml").write_text(
            f"name: {name}\nversion: '1'\ntitle: x\nextends: {parent}\n"
        )
    with pytest.raises(RulesetError, match="循环"):
        load_ruleset("a", root=rules_root)


def test_scenarios_list_skips_shared_bases():
    from engine.rules.loader import list_scenarios

    names = list_scenarios()
    assert "alt_ww2" in names and "_world" not in names
