# vendor/

This directory ships a **frozen subset of [`tinker_cookbook`](https://github.com/thinking-machines-lab/tinker-cookbook)** so that cloning this repo is enough to run the analysis and figure-generation scripts without any additional setup of the cookbook.

## What is included

`vendor/tinker_cookbook/` contains the modules our scripts actually import:

- `__init__.py`, `tokenizer_utils.py`, `model_info.py`, `image_processing_utils.py`
- `renderers/` — chat templates for Qwen3, GPT-OSS, DeepSeek-V3, Kimi, etc.
- `rl/` — minimal RL types/environments needed by `math_env.py`
- `recipes/math_rl/` — the math-grading utilities and `MathEnv`
- top-level utilities used transitively (`completers.py`, `display.py`,
  `cli_utils.py`, `hyperparam_utils.py`, `checkpoint_utils.py`, …)

Tests, unused recipes (`chat_sl/`, `code_rl/`, `harbor_rl/`, …), and the
`reasoning_theater/` recipe (whose contents live at the repo root as
`baee.py`, `experiment.py`, etc.) have been stripped.

## What is NOT included

- The **`tinker` SDK** itself (it requires an API key from
  https://auth.thinkingmachines.ai). Install it via `pip install tinker`.
- `tinker_cookbook` recipes unrelated to math reasoning.

## License

`tinker_cookbook` is released under the Apache License 2.0; the original text
is preserved verbatim in [`LICENSE.tinker_cookbook`](./LICENSE.tinker_cookbook).
Copyright remains with the upstream authors.

## Updating

To refresh against a newer upstream:

```bash
TC=/path/to/tinker-cookbook                       # local clone
DEST=$(pwd)/vendor/tinker_cookbook
rm -rf "$DEST"
rsync -a --exclude='__pycache__' --exclude='*.pyc' --exclude='*_test.py' \
      --exclude='recipes/chat_sl' --exclude='recipes/code_rl' \
      --exclude='recipes/distillation' --exclude='recipes/gpqa' \
      --exclude='recipes/harbor_rl' --exclude='recipes/multiplayer_rl' \
      --exclude='recipes/preference' --exclude='recipes/prompt_distillation' \
      --exclude='recipes/reasoning_theater' --exclude='recipes/rubric' \
      --exclude='recipes/search_tool' --exclude='recipes/verifiers_rl' \
      --exclude='recipes/vlm_classifier' --exclude='chat_app' \
      --exclude='distillation' --exclude='eval' --exclude='example_data' \
      --exclude='preference' --exclude='sandbox' --exclude='scripts' \
      --exclude='supervised' --exclude='tool_use' --exclude='xmux' \
      "$TC/tinker_cookbook/" "$DEST/"
cp "$TC/LICENSE" vendor/LICENSE.tinker_cookbook
```
