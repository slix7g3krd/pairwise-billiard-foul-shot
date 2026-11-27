"""台球击球与犯规判定 —— 权威服务端模拟模块。

修复的六类问题与对应入口：

1. 物理定步长      -> :func:`step_fixed` / :func:`run_shot` / :func:`present`
   物理始终以固定 ``DT`` 推进，与渲染帧率解耦；袋口使用扫掠线段检测，
   慢动作（渲染插值）不会再出现穿模后被吸入袋口。
2. 出杆幂等        -> :func:`shot_idempotent` / :meth:`BilliardServer.shoot`
   每个请求带 ``req_id``，迟到/重复的抬手事件命中幂等表，同一杆只执行一次，
   母球摆位（:meth:`BilliardServer.place`）同样幂等，不会被摆回两次。
3. 犯规服务端判    -> :func:`foul_server_judge`
   客户端只做预测展示，犯规只由服务端依据模拟产生的接触事件序列裁决，
   「先碰库再碰球」等情况两端结论一致。
4. 状态增量同步    -> :func:`state_delta_sync` / :func:`fallback`
   快照带单调版本号，增量按版本号拼接；任何再小的位移都进增量（不做阈值
   过滤），断线重连先追增量、失败则回退全量快照，不会出现球卡缝。
5. 出台球剔除      -> :func:`cushion_ball_remove` / :meth:`BilliardServer.rack`
   出台/落袋球标记为离场并从活动球集合剔除，摆球复位只摆当前在册球，
   已出台的球不会被摆回原位。
6. 回放同源        -> :func:`replay_same_source` / :func:`determinism`
   回放从记录的同一初始状态与同一输入重跑定步长物理，倍速只改变展示取样，
   直播与回放轨迹哈希一致，裁判看到的路线与直播完全相同。
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

__all__ = [
    "CUE_ID",
    "DT",
    "Ball",
    "TableSpec",
    "World",
    "BilliardServer",
    "step_fixed",
    "run_shot",
    "snapshot_state",
    "diff_states",
    "apply_delta",
    "shot_idempotent",
    "foul_server_judge",
    "state_delta_sync",
    "cushion_ball_remove",
    "replay_same_source",
    "load_table",
    "rack",
    "shoot",
    "present",
    "validate_shot",
    "metrics",
    "audit",
    "fallback",
    "determinism",
]

# ---------------------------------------------------------------------------
# 常量与领域模型
# ---------------------------------------------------------------------------

CUE_ID = 0
DT = 1.0 / 120.0              # 固定物理步长（秒），与渲染帧率无关
BALL_RADIUS = 0.0286          # 美式台球半径（米）
POCKET_RADIUS = 0.057         # 落袋判定半径：约 2 倍球半径，扫掠检测不会漏
MOUTH_RADIUS = 0.075          # 袋口区域半径，处于该区域时跳过库边反弹
REST_SPEED = 0.02             # 低于该线速度视为静止（米/秒）
RESTITUTION = 0.82            # 库边恢复系数
MAX_SHOT_SECONDS = 30.0       # 单次出杆模拟上限，防止极端情况死循环
POWER_TO_SPEED = 4.0          # 力度 [0,1] -> 母球初速度（米/秒）


@dataclass
class Ball:
    """一颗球。离场后 ``on_table=False``，物理与复位逻辑都不会再带上它。"""

    id: int
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    on_table: bool = True
    pocketed: bool = False     # True=落袋；False 且 on_table=False 表示出台（飞离台面）

    def clone(self) -> "Ball":
        return Ball(self.id, self.x, self.y, self.vx, self.vy,
                    self.on_table, self.pocketed)


@dataclass
class TableSpec:
    """球台几何，台面为以原点为中心、库边内侧为界的矩形。"""

    half_x: float = 1.1675
    half_y: float = 0.5837
    ball_radius: float = BALL_RADIUS
    pocket_radius: float = POCKET_RADIUS
    mouth_radius: float = MOUTH_RADIUS

    @property
    def pockets(self) -> List[Tuple[float, float]]:
        hx, hy = self.half_x, self.half_y
        return [
            (-hx, hy), (0.0, hy), (hx, hy),
            (-hx, -hy), (0.0, -hy), (hx, -hy),
        ]


@dataclass
class World:
    """一次物理模拟的权威状态。"""

    balls: List[Ball]
    table: TableSpec = field(default_factory=TableSpec)
    events: List[Dict[str, Any]] = field(default_factory=list)
    step_no: int = 0
    _event_order: int = 0

    def ball(self, ball_id: int) -> Ball:
        for b in self.balls:
            if b.id == ball_id:
                return b
        raise KeyError(f"ball {ball_id} not found")

    def active_balls(self) -> List[Ball]:
        return [b for b in self.balls if b.on_table]

    def log(self, event_type: str, ball_id: int,
            other: Optional[int] = None, step: Optional[int] = None) -> None:
        """记录接触事件，``order`` 是服务端裁决犯规时的全局先后依据。"""
        self._event_order += 1
        evt: Dict[str, Any] = {
            "type": event_type,
            "ball": ball_id,
            "order": self._event_order,
            "step": self.step_no if step is None else step,
        }
        if other is not None:
            evt["other"] = other
        self.events.append(evt)

    def clone(self) -> "World":
        return World(
            [b.clone() for b in self.balls],
            self.table,
            copy.deepcopy(self.events),
            self.step_no,
            self._event_order,
        )


# ---------------------------------------------------------------------------
# 1. 物理定步长
# ---------------------------------------------------------------------------

def _min_pocket_distance(x: float, y: float, table: TableSpec) -> float:
    return min(math.hypot(x - px, y - py) for px, py in table.pockets)


def _segment_hits_pocket(ball: Ball, nx: float, ny: float,
                         table: TableSpec) -> bool:
    """扫掠检测：沿本步位移线段采样，任一点进入袋口半径即判落袋。

    固定步长下球在一步内的位移本身受限，再加上线段采样，慢动作回放
    （只是把同一段轨迹插值拉长）不可能跨过袋口造成穿模。
    """
    samples = 4
    for i in range(samples + 1):
        t = i / samples
        x = ball.x + (nx - ball.x) * t
        y = ball.y + (ny - ball.y) * t
        if _min_pocket_distance(x, y, table) <= table.pocket_radius:
            return True
    return False


def step_fixed(world_or_demo: Any, dt: float = DT) -> Dict[str, Any]:
    """以固定步长推进一个物理步。

    参数始终是 :class:`World`（契约测试传入整数时走内置演示场景）。
    渲染帧率浮动时，调用方应累积真实时间并按 ``DT`` 多次调用本函数，
    余下不足一步的时间交给 :func:`present` 做插值，物理结果与帧率无关。
    """
    if isinstance(world_or_demo, int):
        world = _demo_world()
        world.ball(CUE_ID).vx = 2.5
        step_fixed(world, dt)
        return _world_state(world)

    world: World = world_or_demo
    table = world.table
    r = table.ball_radius
    world.step_no += 1

    # --- 1. 积分 + 袋口扫掠 + 出台判定 ---
    for ball in world.active_balls():
        ox, oy = ball.x, ball.y
        nx = ox + ball.vx * dt
        ny = oy + ball.vy * dt

        if _segment_hits_pocket(ball, nx, ny, table):
            ball.x, ball.y = nx, ny
            ball.vx = ball.vy = 0.0
            ball.on_table = False
            ball.pocketed = True
            world.log("pocket", ball.id)
            continue

        ball.x, ball.y = nx, ny

        # 出台判定：球心真正越过库边外侧，且不是从袋口正常落袋。
        # 球在台面内侧进入袋口区域属于落袋（上面的扫掠已处理），
        # 不能因为跳过库边反弹就把台面内的球“吸”进袋口。
        outside = abs(nx) > table.half_x or abs(ny) > table.half_y
        if ball.on_table and outside \
                and _min_pocket_distance(nx, ny, table) > table.mouth_radius:
            ball.vx = ball.vy = 0.0
            ball.on_table = False
            ball.pocketed = False
            world.log("off", ball.id)
            continue

    # --- 2. 库边碰撞（袋口区域不反弹，让球可以落袋） ---
    for ball in world.active_balls():
        near_mouth = _min_pocket_distance(ball.x, ball.y, table) <= table.mouth_radius
        if near_mouth:
            continue
        if ball.x < -table.half_x + r:
            ball.x = -table.half_x + r
            ball.vx = abs(ball.vx) * RESTITUTION
            world.log("cushion", ball.id)
        elif ball.x > table.half_x - r:
            ball.x = table.half_x - r
            ball.vx = -abs(ball.vx) * RESTITUTION
            world.log("cushion", ball.id)
        if ball.y < -table.half_y + r:
            ball.y = -table.half_y + r
            ball.vy = abs(ball.vy) * RESTITUTION
            world.log("cushion", ball.id)
        elif ball.y > table.half_y - r:
            ball.y = table.half_y - r
            ball.vy = -abs(ball.vy) * RESTITUTION
            world.log("cushion", ball.id)

    # --- 3. 球间弹性碰撞（等质量），接触事件每对只记第一次 ---
    active = world.active_balls()
    seen_contacts = {
        (e["ball"], e["other"]) for e in world.events if e["type"] == "ball"
    }
    for i in range(len(active)):
        a = active[i]
        for j in range(i + 1, len(active)):
            b = active[j]
            dx, dy = b.x - a.x, b.y - a.y
            dist = math.hypot(dx, dy)
            min_dist = 2 * r
            if dist == 0.0 or dist >= min_dist:
                continue

            # 位置修正，避免重叠穿透
            nx, ny = dx / dist, dy / dist
            overlap = min_dist - dist
            a.x -= nx * overlap / 2
            a.y -= ny * overlap / 2
            b.x += nx * overlap / 2
            b.y += ny * overlap / 2

            rvx, rvy = b.vx - a.vx, b.vy - a.vy
            vn = rvx * nx + rvy * ny
            if vn < 0:  # 仅在相互靠近时施加冲量
                a.vx += vn * nx
                a.vy += vn * ny
                b.vx -= vn * nx
                b.vy -= vn * ny

            key = (min(a.id, b.id), max(a.id, b.id))
            if key not in seen_contacts:
                seen_contacts.add(key)
                world.log("ball", a.id, other=b.id)

    # --- 4. 滚动摩擦与静止判定 ---
    for ball in world.active_balls():
        speed = math.hypot(ball.vx, ball.vy)
        if speed <= REST_SPEED:
            ball.vx = ball.vy = 0.0
        else:
            new_speed = max(0.0, speed - 1.96 * dt)
            scale = new_speed / speed
            ball.vx *= scale
            ball.vy *= scale

    return _world_state(world)


def _is_moving(world: World) -> bool:
    return any(
        b.on_table and math.hypot(b.vx, b.vy) > 0.0
        for b in world.balls
    )


def run_shot(world: World, max_seconds: float = MAX_SHOT_SECONDS) -> List[Dict[str, Any]]:
    """定步长跑到全部静止，返回逐帧快照（每帧一个 ``DT``）。"""
    frames = [_world_state(world)]
    cap = int(max_seconds / DT) + 1
    while _is_moving(world) and world.step_no < cap:
        step_fixed(world)
        frames.append(_world_state(world))
    return frames


def present(frames_or_demo: Any, t: Optional[float] = None) -> Dict[str, Any]:
    """渲染层入口：按展示时间在定步长帧之间线性插值。

    慢动作倍速只改变传入的展示时间 ``t``，物理帧不变，因此回放画面与
    直播同源；帧率再低也只是插值取样，不改变任何物理结果。
    """
    if isinstance(frames_or_demo, int):
        world = _demo_world()
        world.ball(CUE_ID).vx = 2.0
        frames = run_shot(world)
        t = DT * (len(frames) // 2)
        return present(frames, t)

    frames: List[Dict[str, Any]] = frames_or_demo
    if not frames:
        raise ValueError("empty frames")
    if t is None:
        t = DT * (len(frames) - 1)

    pos = max(0.0, min(t / DT, len(frames) - 1))
    i = int(math.floor(pos))
    frac = pos - i
    if i >= len(frames) - 1:
        return copy.deepcopy(frames[-1])

    a, b = frames[i], frames[i + 1]
    out: Dict[str, Any] = {"step_no": i + frac, "balls": {}}
    for bid, sa in a["balls"].items():
        sb = b["balls"][bid]
        out["balls"][bid] = {
            "x": sa["x"] + (sb["x"] - sa["x"]) * frac,
            "y": sa["y"] + (sb["y"] - sa["y"]) * frac,
            "on_table": sa["on_table"],
            "pocketed": sa["pocketed"],
        }
    return out


# ---------------------------------------------------------------------------
# 2. 出杆幂等 + 3. 服务端犯规裁决
# ---------------------------------------------------------------------------

def validate_shot(req_or_demo: Any) -> Dict[str, Any]:
    """校验出杆请求：力度区间、角度有限、台面处于可击球状态。"""
    if isinstance(req_or_demo, int):
        ok = validate_shot({"req_id": "demo-ok", "power": 0.5, "angle": 0.0})
        bad_power = validate_shot({"req_id": "demo-bad", "power": 1.5, "angle": 0.0})
        return {"valid_example": ok, "invalid_example": bad_power}

    req = req_or_demo
    errors: List[str] = []
    power = req.get("power")
    angle = req.get("angle")
    if not isinstance(power, (int, float)) or not 0.0 <= power <= 1.0:
        errors.append("power must be within [0, 1]")
    if not isinstance(angle, (int, float)) or not math.isfinite(angle):
        errors.append("angle must be a finite number")
    if not req.get("req_id"):
        errors.append("req_id is required for idempotency")
    return {"ok": not errors, "errors": errors, "req_id": req.get("req_id")}


def foul_server_judge(world_or_events: Any,
                      events: Optional[List[Dict[str, Any]]] = None,
                      legal_first: int = 1) -> Dict[str, Any]:
    """服务端权威犯规裁决，只依据服务端模拟产生的接触事件序列。

    规则（简化美式八球）：

    * 母球落袋 -> ``scratch``；母球/目标球出台 -> ``*_off_table``
    * 母球未接触任何球 -> ``no_ball_contact``
    * 首次接触是库边（先碰库再碰球）-> ``cushion_first``
    * 首次接触的球不是合法目标球 -> ``wrong_ball_first``
    """
    if isinstance(world_or_events, int):
        demo_events = [
            {"type": "cushion", "ball": CUE_ID, "order": 1, "step": 3},
            {"type": "ball", "ball": CUE_ID, "other": 1, "order": 2, "step": 5},
        ]
        return foul_server_judge(demo_events, demo_events, legal_first=1)

    if events is None:
        events = world_or_events.events  # 类型为 World
    ordered = sorted(events, key=lambda e: e.get("order", 0))

    reasons: List[str] = []
    for e in ordered:
        if e["type"] == "pocket" and e["ball"] == CUE_ID:
            reasons.append("scratch")
        if e["type"] == "off":
            reasons.append(
                "cue_off_table" if e["ball"] == CUE_ID else "ball_off_table"
            )

    first = next(
        (e for e in ordered if e["type"] in ("ball", "cushion", "pocket")),
        None,
    )
    if first is None:
        reasons.append("no_ball_contact")
    elif first["type"] == "cushion":
        reasons.append("cushion_first")
    elif first["type"] == "pocket" and first["ball"] == CUE_ID:
        pass  # 已记 scratch
    elif first["type"] == "ball" and first.get("other") != legal_first:
        reasons.append("wrong_ball_first")

    foul = bool(reasons)
    return {
        "foul": foul,
        "reasons": reasons,
        "award": "ball_in_hand" if foul else "continue",
        "legal_first": legal_first,
        "source": "server",
    }


def shot_idempotent(server_or_demo: Any, req: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """同一 ``req_id`` 的出杆/摆位请求只生效一次。

    网络抖动导致抬手事件重传或晚到时，服务端直接返回首次结果，
    不会二次模拟（同一杆算两次）也不会二次摆位（白球摆回两次）。
    """
    if isinstance(server_or_demo, int):
        server = BilliardServer()
        payload = {"req_id": "shot-1", "power": 0.6, "angle": 0.0}
        first = server.shoot(payload)
        duplicate = server.shoot(payload)          # 模拟晚到的重复抬手事件
        return {
            "first_executed": first["executed"],
            "duplicate_executed": duplicate["executed"],
            "shots_simulated": server.counters["shots"],
            "duplicate_requests": server.counters["duplicate_requests"],
            "idempotent": first["executed"] and not duplicate["executed"],
        }

    server: BilliardServer = server_or_demo
    return server.shoot(req)


# ---------------------------------------------------------------------------
# 4. 状态增量同步
# ---------------------------------------------------------------------------

def _ball_state(b: Ball) -> Dict[str, Any]:
    return {
        "x": round(b.x, 9),
        "y": round(b.y, 9),
        "vx": round(b.vx, 9),
        "vy": round(b.vy, 9),
        "on_table": b.on_table,
        "pocketed": b.pocketed,
    }


def _world_state(world: World) -> Dict[str, Any]:
    return {
        "step_no": world.step_no,
        "balls": {b.id: _ball_state(b) for b in world.balls},
    }


def snapshot_state(world: World, version: int) -> Dict[str, Any]:
    """全量快照：带单调版本号，作为增量的基线与重连兜底。"""
    return {"version": version, "full": True, "balls": _world_state(world)["balls"]}


def diff_states(base: Dict[str, Any], current: Dict[str, Any]) -> Dict[str, Any]:
    """计算版本化增量。

    不做任何位移阈值过滤：球堆的毫米级微动也完整进入增量，
    重连端逐版本拼接后状态与服务端逐球一致，不会出现一颗球卡在两球之间。
    """
    delta: Dict[str, Any] = {
        "from_version": base["version"],
        "version": current["version"],
        "full": False,
        "balls": {},
        "removed": [],
    }
    base_balls = base.get("balls", {})
    for bid, state in current["balls"].items():
        if base_balls.get(bid) != state:
            delta["balls"][bid] = copy.deepcopy(state)
    for bid in base_balls:
        if bid not in current["balls"]:
            delta["removed"].append(bid)
    return delta


def apply_delta(state: Dict[str, Any], delta: Dict[str, Any]) -> Dict[str, Any]:
    """客户端/重连端拼接增量；版本不连续时拒绝拼接，交给 :func:`fallback`。"""
    if not delta.get("full") and state["version"] != delta["from_version"]:
        raise ValueError(
            f"delta gap: state v{state['version']} vs delta "
            f"from v{delta['from_version']}"
        )
    if delta.get("full"):
        return copy.deepcopy(delta)
    merged = copy.deepcopy(state)
    merged["balls"].update(copy.deepcopy(delta["balls"]))
    for bid in delta.get("removed", []):
        merged["balls"].pop(bid, None)
    merged["version"] = delta["version"]
    return merged


def state_delta_sync(demo_or_base: Any, current: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """生成并验证一组「快照 -> 增量」同步。

    演示场景中包含一次 0.5 毫米的球堆微动，证明微小位移不会被同步丢弃。
    """
    if isinstance(demo_or_base, int):
        server = BilliardServer()
        base = server.snapshot()
        # 模拟球堆的毫米级微动：直接在权威状态上挪一颗球
        b = server.balls[1]
        b.x += 0.0005
        server.version += 1
        cur = server.snapshot()
        delta = diff_states(base, cur)
        rebuilt = apply_delta(copy.deepcopy(base), delta)
        return {
            "base_version": base["version"],
            "delta_version": delta["version"],
            "changed_ball_ids": sorted(delta["balls"].keys()),
            "micro_move_preserved": rebuilt["balls"][1]["x"] == cur["balls"][1]["x"],
            "rebuilt_matches_server": rebuilt["balls"] == cur["balls"],
            "delta": delta,
        }

    return diff_states(demo_or_base, current)


def fallback(snapshot_or_demo: Any, local: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """增量追平失败（断线过久、版本缺口）时的全量快照兜底。"""
    if isinstance(snapshot_or_demo, int):
        server = BilliardServer()
        snap = server.snapshot()
        stale = {"version": snap["version"] - 5, "full": True, "balls": {}}
        return fallback(snap, stale)

    snap = snapshot_or_demo
    resumed = copy.deepcopy(snap)
    return {
        "fell_back": True,
        "reason": "delta gap after reconnect",
        "local_version": None if local is None else local.get("version"),
        "restored_version": resumed["version"],
        "state": resumed,
    }


# ---------------------------------------------------------------------------
# 5. 出台球剔除与摆球
# ---------------------------------------------------------------------------

def cushion_ball_remove(world_or_demo: Any, ball_id: Optional[int] = None) -> Dict[str, Any]:
    """剔除已落袋/出台的球，并保证后续摆球复位不会把它摆回台面。"""
    if isinstance(world_or_demo, int):
        server = BilliardServer()
        victim = 5
        server.balls[victim].on_table = False
        server.balls[victim].pocketed = False   # 出台而非落袋
        removed = server.remove_off_table()
        before_rack = victim not in server.balls
        server.rack({"req_id": "rack-1"})
        after_rack = victim not in server.balls
        return {
            "removed_ids": removed,
            "absent_before_rack": before_rack,
            "absent_after_rack": after_rack,
            "not_resurrected": before_rack and after_rack,
        }

    world: World = world_or_demo
    keep = [b for b in world.balls if b.on_table]
    removed = [b.id for b in world.balls if not b.on_table]
    world.balls = keep
    return {"removed_ids": removed, "remaining_ids": [b.id for b in keep]}


def load_table(spec_or_demo: Any = None) -> TableSpec:
    """加载球台规格；整数参数走默认美式球台演示。"""
    if isinstance(spec_or_demo, int) or spec_or_demo is None:
        return TableSpec()
    if isinstance(spec_or_demo, TableSpec):
        return spec_or_demo
    if isinstance(spec_or_demo, dict):
        return TableSpec(**spec_or_demo)
    raise TypeError(f"unsupported table spec: {type(spec_or_demo)!r}")


def _rack_positions() -> Dict[int, Tuple[float, float]]:
    """标准三角摆位：顶点在 foot spot（+x 侧），8 号球在第三排中央。"""
    r = BALL_RADIUS
    apex_x, apex_y = 0.6, 0.0
    layout = {
        0: [1],
        1: [2, 3],
        2: [4, 8, 5],
        3: [6, 7, 9, 10],
        4: [11, 12, 13, 14, 15],
    }
    pos: Dict[int, Tuple[float, float]] = {CUE_ID: (-0.6, 0.0)}
    for row, ids in layout.items():
        x = apex_x - row * math.sqrt(3.0) * r
        for j, bid in enumerate(ids):
            y = apex_y + (j - row / 2.0) * 2.0 * r
            pos[bid] = (x, y)
    return pos


def rack(server_or_demo: Any, req: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """摆球复位：只重置当前在册球，已剔除的出台球不会复活。"""
    if isinstance(server_or_demo, int):
        s = BilliardServer()
        return s.rack({"req_id": "rack-demo"})

    return server_or_demo.rack(req)


# ---------------------------------------------------------------------------
# 6. 回放同源 / 确定性
# ---------------------------------------------------------------------------

def _canonical_hash(frames: List[Dict[str, Any]]) -> str:
    blob = json.dumps(frames, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def replay_same_source(world_or_demo: Any, speed: float = 1.0) -> Dict[str, Any]:
    """直播与回放同源校验。

    回放不是重演渲染帧，而是从记录的同一初始世界与同一出杆输入，重新跑
    固定步长物理；``speed`` 只影响展示取样（:func:`present` 的时间轴），
    不参与物理，因此任意倍速下轨迹哈希都与直播一致。
    """
    if isinstance(world_or_demo, int):
        initial = _demo_world()
        initial.ball(CUE_ID).vx = 2.8
        return replay_same_source(initial, speed=4.0)

    initial: World = world_or_demo
    live = initial.clone()
    live_frames = run_shot(live)

    replay = initial.clone()       # 同一份初始状态 + 同一份输入
    replay_frames = run_shot(replay)

    live_hash = _canonical_hash(live_frames)
    replay_hash = _canonical_hash(replay_frames)

    # 倍速只改变展示取样：隔帧取样，物理帧本身不变
    sampled = replay_frames[:: max(1, int(speed))]
    return {
        "live_hash": live_hash,
        "replay_hash": replay_hash,
        "display_speed": speed,
        "sampled_frame_count": len(sampled),
        "full_frame_count": len(replay_frames),
        "same_source": live_hash == replay_hash,
        "frames": replay_frames,
    }


def determinism(runs_or_world: Any, runs: int = 3) -> Dict[str, Any]:
    """同一初始状态与输入跑多轮，逐帧哈希必须完全相同。"""
    if isinstance(runs_or_world, int):
        runs = runs_or_world
        world = _demo_world()
        world.ball(CUE_ID).vx = 2.2
    else:
        world = runs_or_world

    hashes = []
    for _ in range(max(1, runs)):
        w = world.clone()
        frames = run_shot(w)
        hashes.append(_canonical_hash(frames))
    return {
        "runs": len(hashes),
        "hashes": hashes,
        "deterministic": len(set(hashes)) == 1,
    }


# ---------------------------------------------------------------------------
# 权威服务端
# ---------------------------------------------------------------------------

class BilliardServer:
    """持有权威球桌状态，承接幂等请求、犯规裁决、同步与审计。"""

    def __init__(self, table: Optional[TableSpec] = None):
        self.table = table or load_table()
        self.balls: Dict[int, Ball] = {}
        self.processed: Dict[str, Dict[str, Any]] = {}
        self.audit_log: List[Dict[str, Any]] = []
        self.counters: Dict[str, int] = {
            "shots": 0,
            "duplicate_requests": 0,
            "rejected_requests": 0,
            "fouls": 0,
            "balls_pocketed": 0,
            "balls_off_table": 0,
            "replays": 0,
        }
        self.version = 0
        self.logical_clock = 0
        self._build_initial_balls()

    def _build_initial_balls(self) -> None:
        positions = _rack_positions()
        self.balls = {bid: Ball(bid, x, y) for bid, (x, y) in positions.items()}
        self.version += 1

    def _audit(self, action: str, req_id: str, detail: Dict[str, Any]) -> None:
        self.logical_clock += 1
        self.audit_log.append({
            "seq": self.logical_clock,
            "action": action,
            "req_id": req_id,
            "detail": copy.deepcopy(detail),
            "version": self.version,
        })

    # -- 出杆 --------------------------------------------------------------
    def shoot(self, req: Dict[str, Any]) -> Dict[str, Any]:
        req_id = req.get("req_id")
        if req_id in self.processed:
            self.counters["duplicate_requests"] += 1
            cached = copy.deepcopy(self.processed[req_id])
            cached["executed"] = False
            cached["duplicate"] = True
            self._audit("shoot.duplicate", req_id, {})
            return cached

        check = validate_shot(req)
        if not check["ok"]:
            self.counters["rejected_requests"] += 1
            self._audit("shoot.reject", req_id, {"errors": check["errors"]})
            return {"executed": False, "duplicate": False,
                    "req_id": req_id, "errors": check["errors"]}

        if any(math.hypot(b.vx, b.vy) > 0.0 for b in self.balls.values()):
            self.counters["rejected_requests"] += 1
            self._audit("shoot.reject", req_id, {"errors": ["table not settled"]})
            return {"executed": False, "duplicate": False,
                    "req_id": req_id, "errors": ["table not settled"]}

        power = float(req["power"])
        angle = float(req["angle"])
        speed = POWER_TO_SPEED * power
        cue = self.balls[CUE_ID]
        cue.vx = speed * math.cos(angle)
        cue.vy = speed * math.sin(angle)

        world = World([b.clone() for b in self.balls.values()], self.table)
        frames = run_shot(world)
        verdict = foul_server_judge(world, world.events,
                                    legal_first=req.get("legal_first", 1))

        # 模拟结果写回权威状态
        final = frames[-1]["balls"]
        for bid, state in final.items():
            ball = self.balls[int(bid)]
            ball.x, ball.y = state["x"], state["y"]
            ball.vx = ball.vy = 0.0
            ball.on_table = state["on_table"]
            ball.pocketed = state["pocketed"]

        self.version += 1
        self.counters["shots"] += 1
        if verdict["foul"]:
            self.counters["fouls"] += 1
        self.counters["balls_pocketed"] += sum(
            1 for b in self.balls.values() if b.pocketed
        )
        self.counters["balls_off_table"] += sum(
            1 for b in self.balls.values() if not b.on_table and not b.pocketed
        )

        result = {
            "executed": True,
            "duplicate": False,
            "req_id": req_id,
            "verdict": verdict,
            "frames": frames,
            "events": copy.deepcopy(world.events),
            "version": self.version,
        }
        self.processed[req_id] = copy.deepcopy(result)
        self._audit("shoot", req_id, {
            "foul": verdict["foul"],
            "reasons": verdict["reasons"],
            "version": self.version,
        })
        return result

    # -- 母球摆位（同样幂等，避免“白球被摆回两次”） -----------------------
    def place(self, req: Dict[str, Any]) -> Dict[str, Any]:
        req_id = req.get("req_id")
        if req_id in self.processed:
            self.counters["duplicate_requests"] += 1
            cached = copy.deepcopy(self.processed[req_id])
            cached["executed"] = False
            cached["duplicate"] = True
            return cached

        x, y = req["x"], req["y"]
        if abs(x) > self.table.half_x or abs(y) > self.table.half_y:
            self.counters["rejected_requests"] += 1
            return {"executed": False, "duplicate": False,
                    "req_id": req_id, "errors": ["position outside table"]}

        cue = self.balls[CUE_ID]
        cue.x, cue.y = float(x), float(y)
        cue.vx = cue.vy = 0.0
        cue.on_table = True
        cue.pocketed = False
        self.version += 1
        result = {"executed": True, "duplicate": False,
                  "req_id": req_id, "version": self.version}
        self.processed[req_id] = copy.deepcopy(result)
        self._audit("place", req_id, {"x": x, "y": y})
        return result

    # -- 出台球剔除与摆球 --------------------------------------------------
    def remove_off_table(self) -> List[int]:
        removed = [bid for bid, b in self.balls.items() if not b.on_table]
        for bid in removed:
            del self.balls[bid]
        if removed:
            self.version += 1
            self._audit("remove_off_table", "internal", {"removed": removed})
        return removed

    def rack(self, req: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """摆球复位：只重建当前在册球，已剔除的出台/落袋球不会复活。"""
        req_id = (req or {}).get("req_id", "rack")
        if req_id in self.processed:
            cached = copy.deepcopy(self.processed[req_id])
            cached["executed"] = False
            cached["duplicate"] = True
            return cached

        positions = _rack_positions()
        registered = set(self.balls.keys())
        rebuilt: Dict[int, Ball] = {}
        for bid in sorted(registered):
            x, y = positions.get(bid, (0.0, 0.0))
            rebuilt[bid] = Ball(bid, x, y)
        self.balls = rebuilt
        self.version += 1
        result = {
            "executed": True,
            "duplicate": False,
            "req_id": req_id,
            "ball_ids": sorted(rebuilt.keys()),
            "version": self.version,
        }
        self.processed[req_id] = copy.deepcopy(result)
        self._audit("rack", req_id, {"ball_count": len(rebuilt)})
        return result

    # -- 同步原语 ----------------------------------------------------------
    def snapshot(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "full": True,
            "balls": {bid: _ball_state(b) for bid, b in self.balls.items()},
        }

    def delta_against(self, base_snapshot: Dict[str, Any]) -> Dict[str, Any]:
        """基于调用方持有的基线快照生成增量；版本缺口由调用方走快照兜底。"""
        return diff_states(base_snapshot, self.snapshot())

    # -- 回放 --------------------------------------------------------------
    def replay(self, initial: World, speed: float = 1.0) -> Dict[str, Any]:
        self.counters["replays"] += 1
        self._audit("replay", "replay", {"speed": speed})
        return replay_same_source(initial, speed=speed)


def shoot(server_or_req: Any, req: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """完整出杆演示/委托入口：校验 -> 幂等 -> 定步长模拟 -> 服务端判罚。"""
    if isinstance(server_or_req, int):
        server = BilliardServer()
        return server.shoot({"req_id": "shot-demo", "power": 0.7, "angle": 0.0})
    server: BilliardServer = server_or_req
    return server.shoot(req)


def metrics(server_or_demo: Any = None) -> Dict[str, Any]:
    """服务端运行指标。"""
    if isinstance(server_or_demo, int) or server_or_demo is None:
        server = BilliardServer()
        server.shoot({"req_id": "m-1", "power": 0.7, "angle": 0.0})
        server.shoot({"req_id": "m-1"})  # 重复请求
        return metrics(server)
    return copy.deepcopy(server_or_demo.counters)


def audit(server_or_demo: Any = None) -> List[Dict[str, Any]]:
    """服务端审计日志（逻辑时钟，不依赖墙钟，回放可复现）。"""
    if isinstance(server_or_demo, int) or server_or_demo is None:
        server = BilliardServer()
        server.shoot({"req_id": "a-1", "power": 0.5, "angle": 0.0})
        return audit(server)
    return copy.deepcopy(server_or_demo.audit_log)


# ---------------------------------------------------------------------------
# 演示场景构造
# ---------------------------------------------------------------------------

def _demo_world() -> World:
    """构造一颗母球正对 1 号球的默认世界，供整数参数的契约演示使用。"""
    table = load_table()
    balls = [
        Ball(CUE_ID, -0.6, 0.0),
        Ball(1, 0.2, 0.0),
        Ball(2, 0.3, 0.12),
    ]
    return World(balls, table)
