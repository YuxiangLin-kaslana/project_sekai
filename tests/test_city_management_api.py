from fastapi.testclient import TestClient

from server.app import app
from server.game import unpack

client = TestClient(app)


def test_city_management_and_direct_construction_api():
    snap = client.post(
        "/api/games", json={"seed": 5, "width": 40, "height": 25, "nations": 4}
    ).json()
    path = "/api/games/" + snap["id"]
    assert not any(n["city_management"] for n in snap["nations"])
    assert set(snap["rules"]["city_policies"]) == {
        "balanced",
        "growth",
        "industry",
        "wealth",
        "military",
    }
    city = snap["cities"][0]
    options = client.get(path + "/city-options", params={"city": city["id"]}).json()
    assert options["state_hash"] == snap["state_hash"] and options["options"]
    option = next(o for o in options["options"] if "farm" in o["facilities"])
    before_units = [(u["id"], u["tile"]) for u in snap["units"]]
    action = client.post(
        path + "/actions",
        json={
            "expected_hash": snap["state_hash"],
            "orders": [
                {
                    "issuer": city["nation"],
                    "type": "develop_city",
                    "city": city["id"],
                    "facility": "farm",
                    "to": option["to"],
                }
            ],
        },
    ).json()
    assert not action["rejected"] and action["turn"] == snap["turn"]
    c = next(c for c in action["cities"] if c["id"] == city["id"])
    assert c["construction"]["turns_left"] == 3 and not c["managed"]
    # 工期 3 回合起，造价按期从本城生产力里付，付清才竣工
    for _ in range(30):
        action = client.post(path + "/turn", json={"autoplay": False}).json()
        c = next(c for c in action["cities"] if c["id"] == city["id"])
        if c["construction"] is None:
            break
    assert [(u["id"], u["tile"]) for u in action["units"]] == before_units
    assert c["construction"] is None and unpack(action["facility"])[option["tile"]] > 0
    enabled = client.post(
        path + "/actions",
        json={
            "expected_hash": action["state_hash"],
            "orders": [
                {
                    "type": "set_policy",
                    "issuer": city["nation"],
                    "enabled": True,
                    "policy": "industry",
                },
                {
                    "type": "city_management",
                    "issuer": city["nation"],
                    "city": city["id"],
                    "enabled": True,
                },
            ],
        },
    ).json()
    assert enabled["turn"] == action["turn"] and not enabled["rejected"]
    managed = client.post(path + "/turn", json={"autoplay": False}).json()
    n = next(n for n in managed["nations"] if n["id"] == city["nation"])
    assert n["city_management"] and n["policy"] == "industry" and n["management"]["reserve"] >= 20
    assert any(
        e["type"] == "directive_report" and e["status"] == "started" for e in managed["events"]
    )
    assert client.get(path + "/city-options", params={"city": "city:999999"}).status_code == 404
    assert client.get(path + "/city-options", params={"city": "invalid"}).status_code == 422
