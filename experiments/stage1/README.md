# Stage I — first agent loop

This directory contains the deterministic environment, a minimal DeepSeek client, and a one-episode runner for **control mode A**. This first runner is deliberately stateless across decisions and episodes: each call sends only the current public observation and the available-action list to the model. It does not implement external memory, retrieval, or rethink yet.

## Prerequisites

- Python 3.10 or newer.
- pytest for tests.
- A DeepSeek API key kept in a local, ignored .env file or exported in the shell. The key is never written to the log.

Load a local .env in a Bash shell:

```bash
set -a
source .env
set +a
```

Run tests from the repository root:

```bash
pytest -q experiments/stage1
```

## Run one baseline episode

Use a new log path outside the repository so research records are not accidentally committed:

```bash
python -m experiments.stage1.run_episode \
  --seed 1234 \
  --log /tmp/construct-stage1-A-seed1234.jsonl
```

The runner uses deepseek-flash by default and requires no additional runtime Python package. A different model can be selected with --model. To test the cost-law reversal on the same map, add --swapped-law; this keeps the seed-generated geometry fixed while swapping the hidden A/B movement costs.

The output path must not already exist. This prevents an experiment from silently overwriting an earlier run.

## Journal and information boundary

The JSONL journal contains:

- the public observation and available actions seen by the model;
- the action, short expectation, model output, objective result, and next observation;
- token usage and terminal result;
- researcher-only map and hidden cost-law fields for replay.

**The research-only fields are never sent to the model.** Treat the entire JSONL file as an experimenter artifact, not as an agent input. Never put the API key in the journal.

This is a control-A baseline only. Modes B and C (persistent experience and continuous retrieval/rethink), plus aggregate comparison across fixed episodes, should be added after the first run and its journal have been inspected. A win in a single episode is not evidence for the Construct hypothesis.
