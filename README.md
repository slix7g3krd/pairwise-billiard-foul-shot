# pairwise-billiard-foul-shot

## 领域

游戏、图形与多媒体

## 主题

台球击球与犯规判定

## 说明

物理定步长、出杆幂等与服务端犯规裁决。

初始状态下 `gameplay/billiard_foul_shot.py` 中所有函数抛出 `NotImplementedError`，测试全部失败。
修复实现后运行：

```bash
python -m pytest -q
```

期望全部通过。
