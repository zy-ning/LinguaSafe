# LinguaSafe

A multilingual LLM safety evaluation framework. Evaluates models against unsafe content across 12 languages and 5 harm categories, with support for jailbreak robustness testing.

**Languages:** English, Chinese, Arabic, Russian, Serbian, Thai, Korean, Vietnamese, Czech, Hungarian, Bengali, Malay

**Categories:** Explicit Content · Crimes & Illegal Activities · Harm & Misuse · Privacy & Property · Fairness & Justice

## Installation

```bash
uv venv --python 3.10
uv sync
```

Create `api.toml` (already in `.gitignore`):

```toml
[openai]
api_key = "sk-..."
models = ["gpt-4o-2024-11-20", "gpt-4o-mini"]

[local]
base_url = "http://localhost:8000/v1"
models = ["Llama-3.1-8B-Instruct", "Qwen2.5-7B-Instruct"]

[oai_moderate]
api_key = "sk-..."
base_url = "https://api.openai-proxy.com/v1"
models = ["omni-moderation-latest"]
```

Each section maps to an API endpoint. `models` lists which model names route to that endpoint.

## Usage

```bash
# Evaluate a model across all languages
python -m linguasafe.eval --model gpt-4o-2024-11-20

# Single language
python -m linguasafe.eval --model gpt-4o-2024-11-20 --lang zh

# Generalization mode (non-language-specific prompts only)
python -m linguasafe.eval --model gpt-4o-2024-11-20 --generalization

# Jailbreak robustness (templates: DAN, AIM, STAN, DUDE, MONGO, EVIL_BOT)
python -m linguasafe.eval --model gpt-4o-2024-11-20 --jailbreak DAN
```

Re-running the same command resumes from where it left off — already-completed tasks are skipped automatically.

### Options

| Flag | Default | Description |
|---|---|---|
| `--model` | `gpt-4o-2024-11-20` | Model under evaluation |
| `--lang` | `all` | ISO language code, or `all` |
| `--generalization` | off | Restrict to non-language-specific prompts |
| `--jailbreak` | — | Jailbreak template to wrap prompts |
| `--assitSLM` | `internlm3-8b-instruct` | SLM used for reject/safety judging |
| `--limit` | `64` | Parallel worker threads |

Results are saved to `logs/res/`.
