import pytest
import gameplay.billiard_foul_shot as mod

CASES = [
    ("step_fixed", (1,)),
    ("shot_idempotent", (1,)),
    ("foul_server_judge", (1,)),
    ("state_delta_sync", (1,)),
    ("cushion_ball_remove", (1,)),
    ("replay_same_source", (1,)),
    ("load_table", (1,)),
    ("rack", (1,)),
    ("shoot", (1,)),
    ("present", (1,)),
    ("validate_shot", (1,)),
    ("metrics", (1,)),
    ("audit", (1,)),
    ("fallback", (1,)),
    ("determinism", (1,)),
]

@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None
