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
models = ["gpt-4o-2024-11-20", "gpt-5-mini"]

[local]
base_url = "http://localhost:8000/v1"
models = ["Llama-3.1-8B-Instruct", "..."]

[oai_moderate]
api_key = "sk-..."
models = ["omni-moderation-latest"]
```

Each section maps to an API endpoint. `models` lists which model names route to that endpoint.

## Usage

```bash
# Evaluate a model across all languages
python -m linguasafe.eval --model gpt-4o-2024-11-20 --assitSLM gpt-5-mini

# Single language
python -m linguasafe.eval --model gpt-4o-2024-11-20 --lang zh
```

Re-running the same command resumes from where it left off — already-completed tasks are skipped automatically.

Results are saved to `logs/res/`.
