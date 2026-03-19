# LinguaSafe


## Installation & Setup

`uv` is the recommended tool for managing dependencies and running commands in this repository. It provides a consistent environment and simplifies the setup process.

```bash
uv venv --python 3.10
uv sync
```


create your `api.toml` based on the provided example, and ensure it is listed in `.gitignore` to prevent accidental commits of sensitive information.
```toml
# Configuration file for the API

[openai]
  api_key = ""
  # base_url = ""
  models = ["gpt-4o-2024-11-20", "gpt-4o-mini", "omni-moderation-latest"]

# [gemini]
# [claude]

[local]
  base_url = "http://localhost:8000/v1"
  models = [
    "Mistral-7B-Instruct-v0.3",
    "Phi-4",
    "Qwen2.5-7B-Instruct",
    "gemma-2-27b-it",
    "Llama-3.1-8B-Instruct",
  ]

```
