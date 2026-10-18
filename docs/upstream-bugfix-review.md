# StackPlanner training branch review

Reviewed on 2026-10-16 against Harness-RL's original tree (`740f24c`).
That tree is identical to the upstream common base (`eacd95be`); the projects
have different Git histories. Changes are selectively ported so the optional
OLE integration remains intact.

| Branch | Reviewed revision | Applicable fixes |
| --- | --- | --- |
| [Search](https://github.com/EunTilofy/StackPlanner-RL-Training/tree/756d681cd2560651c17f628208cc5cf5dd48496e) | `756d681c` | Episode-local recording of Search-R1 infrastructure failures, including failures handled inside frozen specialists; invalidate the episode before rewards or policy-failure draining. |
| [ALFW](https://github.com/EunTilofy/StackPlanner-RL-Training/tree/6cea845a2f9e7757189933a4aaa27bb981ea1ca6) | `6cea845a` | Preserve cancellation during policy-failure draining; log actor cleanup errors without replacing the original failure or cancellation. |
| [WebS](https://github.com/EunTilofy/StackPlanner-RL-Training/tree/352fb1866f7aade66b9fb519730b0b48606eae6d) | `352fb186` | Same active runtime fixes as ALFW. Both branches also retain a greedy evaluation fix in `strategies/interactor_mixed_memory/overlay/slime/slime_plugins/stackplanner/generate.py`; that fix is ported to actor and frozen model builders. |

The evaluation sampling fix was checked against the adapter's actual sampling
precedence: OpenAI request temperature and top-p override session defaults.
Thus evaluation requests use 0/1 even though session defaults retain the
training configuration. Training requests keep their existing values.

Related local fixes ensure aborted samples carry rollout lineage, so OLE can
exclude them without failing its input checks; reject non-object retriever
payloads through the same infrastructure-error path; and let process-exit
signals propagate after actor cleanup.

Task-specific ALFWorld/WebShop workers, datasets, benchmark configuration,
GT/oracle memory strategies, and separate LoRA training support add new
capabilities rather than repair the existing Harness-RL training route.
They are outside this port. The virtualenv interpreter-path fix in the latest
ALFW/WebS commits applies to their new interactive-run materialization script,
which is absent from Harness-RL.

## Regression checks

The lifecycle tests use the real episode model factory and reward code with
mocked graph/model boundaries. They cover hidden retriever outages, policy
failures, concurrent episode isolation, ContextVar restoration, cancellation
during trajectory draining and cleanup, cleanup exceptions, rollout lineage,
OLE exclusion, and effective evaluation sampling in both training modes.
Tool tests cover transport/JSON/schema failures, empty results, bounded error
recording, and context propagation to a specialist thread.

```bash
python -m pytest tests -q -o addopts=''
PYTHONPATH="$PWD/slime" python -m pytest \
  slime/tests/test_reward_normalization.py \
  slime/tests/test_dp_schedule.py -q -o addopts=''
cd StackPlanner2.0/backend
PYTHONPATH="$PWD/packages/harness" python -m pytest \
  tests/test_search_r1_tools.py -q
```

The complete backend suite additionally needs the application's server,
database, channel, and TUI dependencies. The initial training test environment
could not collect that suite (85 collection errors from missing dependencies,
including FastAPI, Blockbuster, and Rich). Selected regression tests do not
constitute a full distributed model-training run.

Validation: 232 training/normalization/scheduling tests passed
with one skipped; all 11 Search-R1 tool tests passed. The OLE policy-gradient
and optimizer tests passed in FP32/BF16 across eight independent GPU workers
(48 GPU checks). Ruff lint/format checks and `git diff --check` passed.
The post-change full-backend attempt still stopped at collection because
Blockbuster was unavailable.
