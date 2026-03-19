import concurrent.futures
import hashlib
import json
import pathlib
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

import tomli
from openai import OpenAI
from misc import PROJECT_DIR


@dataclass
class ClientConfig:
    """Configuration for an API client."""
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    models: List[str] = field(default_factory=list)
    prefix: str = ""


@dataclass
class Task:
    """Represents a single LLM generation task."""
    prompt: Optional[str] = None
    messages: Optional[List[Dict[str, str]]] = None
    model_name: Optional[str] = None
    temperature: float = 1.0
    max_tokens: int = 1024
    top_p: float = 0.95
    extra_params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.prompt and not self.messages:
            raise ValueError("Task must contain either 'prompt' or 'messages'")
        if self.prompt and self.messages:
            raise ValueError("Task cannot contain both 'prompt' and 'messages'")


class LLMClientManager:
    """Manages multiple LLM API clients."""

    def __init__(self, config_path: Optional[pathlib.Path] = None):
        if config_path is None:
            config_path = PROJECT_DIR / "api.toml"

        self.config = self._load_config(config_path)
        self.clients = {}
        self.model_to_client = {}
        self._initialize_clients()

    def _load_config(self, config_path: pathlib.Path) -> Dict:
        """Load configuration from TOML file."""
        with open(config_path, "rb") as f:
            return tomli.load(f)

    def _create_client(self, client_config: ClientConfig) -> OpenAI:
        """Create an OpenAI client."""
        base_params = {
            "api_key": client_config.api_key,
            "base_url": client_config.base_url,
        }

        if client_config.api_key is None:
            base_params.pop("api_key")

        return OpenAI(**base_params)

    def _initialize_clients(self):
        """Initialize all API clients from configuration."""
        # Dynamically use all configurations from api.toml
        for config_key, api_config in self.config.items():
            if not isinstance(api_config, dict):
                continue

            try:
                client_config = ClientConfig(
                    api_key=api_config.get("api_key"),
                    base_url=api_config.get("base_url"),
                    models=api_config.get("models", []),
                    prefix=api_config.get("prefix", "")
                )

                # Only create client if it has either api_key or base_url
                if client_config.api_key or client_config.base_url:
                    self.clients[config_key] = {
                        "client": self._create_client(client_config, None),
                        "config": client_config
                    }

                    # Map models to clients
                    for model in client_config.models:
                        self.model_to_client[model] = config_key

            except Exception as e:
                print(f"Warning: Failed to initialize client for {config_key}: {e}")

    def get_client_for_model(self, model_name: str) -> tuple[OpenAI, str]:
        """Get the appropriate client and model name for a given model."""
        if model_name not in self.model_to_client:
            raise ValueError(f"Unsupported model: {model_name}")

        client_name = self.model_to_client[model_name]
        client_info = self.clients[client_name]
        client = client_info["client"]

        # Apply prefix if needed
        prefix = client_info["config"].prefix
        final_model_name = f"{prefix}{model_name}" if prefix else model_name

        return client, final_model_name


class LLMGenerator:
    """Main class for LLM generation tasks."""

    def __init__(self, config_path: Optional[pathlib.Path] = None):
        self.client_manager = LLMClientManager(config_path)

    def generate(self, task: Task, model_name: Optional[str] = None) -> str:
        """Generate text using the specified model and task parameters."""
        model_name = model_name or task.model_name
        if not model_name:
            raise ValueError("Model name must be specified")

        client, final_model_name = self.client_manager.get_client_for_model(model_name)

        messages = task.messages
        if task.prompt:
            use_system = "gemma" not in model_name.lower()
            messages = self._prompt_to_messages(task.prompt, use_system)

        response = client.chat.completions.create(
            model=final_model_name,
            messages=messages,
            temperature=task.temperature,
            max_tokens=task.max_tokens,
            top_p=task.top_p,
            **task.extra_params,
        )

        return response.choices[0].message.content

    @staticmethod
    def _prompt_to_messages(prompt: str, use_system: bool = True) -> List[Dict[str, str]]:
        """Convert a prompt to a list of messages."""
        messages = [{"role": "user", "content": prompt}]
        if use_system:
            messages.insert(0, {"role": "system", "content": "You are a helpful assistant."})
        return messages


class BatchProcessor:
    """Handles batch processing of LLM generation tasks."""

    def __init__(self, generator: LLMGenerator):
        self.generator = generator
        self.results_lock = threading.Lock()
        self.results = []

    def process_batch(
        self,
        tasks: Union[List[Dict], str],
        output_file: str,
        default_model: str = "gpt-4o-2024-11-20",
        check_func: Optional[Callable[[str], bool]] = None,
        max_retries: int = 5,
        max_workers: int = 16,
    ) -> List[Dict]:
        """Process multiple tasks in parallel with incremental saving and error recovery."""

        # Load tasks
        if isinstance(tasks, str):
            with open(tasks, "r", encoding="utf-8") as f:
                tasks = json.load(f)

        # Convert to Task objects
        task_objects = [self._dict_to_task(task_dict, default_model) for task_dict in tasks]

        output_path = pathlib.Path(output_file)
        output_path.touch(exist_ok=True)

        # Load existing results and determine what needs to be processed
        completed_hashes, existing_results = self._load_existing_results(output_path)
        task_hashes = [self._generate_task_hash(task) for task in task_objects]

        tasks_to_fix = self._identify_error_tasks(task_objects, task_hashes, existing_results)
        tasks_to_run = self._identify_new_tasks(task_objects, task_hashes, completed_hashes)

        # Process tasks
        if tasks_to_fix:
            print(f"Fixing {len(tasks_to_fix)} error tasks...")
            self._process_tasks(tasks_to_fix, output_path, check_func, max_retries, max_workers, is_fix=True)

        if tasks_to_run:
            print(f"Processing {len(tasks_to_run)} new tasks...")
            self._process_tasks(tasks_to_run, output_path, check_func, max_retries, max_workers)

        # Generate final results
        final_results = self._compile_final_results(task_objects, task_hashes, existing_results)
        self._save_final_results(final_results, output_path.with_suffix(".res.jsonl"))

        print(f"Batch processing completed. {len(self.results)} tasks processed.")
        return final_results

    def _dict_to_task(self, task_dict: Dict, default_model: str) -> Task:
        """Convert dictionary to Task object."""
        extra_params = {
            k: v for k, v in task_dict.items()
            if k not in ["prompt", "messages", "model_name", "temperature", "max_tokens", "top_p"]
        }

        return Task(
            prompt=task_dict.get("prompt"),
            messages=task_dict.get("messages"),
            model_name=task_dict.get("model_name", default_model),
            temperature=task_dict.get("temperature", 1.0),
            max_tokens=task_dict.get("max_tokens", 1024),
            top_p=task_dict.get("top_p", 0.95),
            extra_params=extra_params
        )

    def _generate_task_hash(self, task: Task) -> str:
        """Generate a unique hash for a task."""
        use_system = "gemma" not in (task.model_name or "").lower()

        if task.prompt:
            content = json.dumps(self.generator._prompt_to_messages(task.prompt, use_system))
        else:
            content = json.dumps(task.messages)

        return hashlib.md5(content.encode("utf-8")).hexdigest()

    def _load_existing_results(self, output_path: pathlib.Path) -> tuple[set, list]:
        """Load existing results from output file."""
        completed_hashes = set()
        existing_results = []

        if not output_path.exists():
            return completed_hashes, existing_results

        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    result = json.loads(line)
                    if "task_hash" in result:
                        completed_hashes.add(result["task_hash"])
                    existing_results.append(result)
                except json.JSONDecodeError:
                    print(f"Skipping invalid JSON line: {line.strip()}")

        return completed_hashes, existing_results

    def _identify_error_tasks(self, tasks: List[Task], task_hashes: List[str], existing_results: List[Dict]) -> List[tuple]:
        """Identify tasks that failed and need to be retried."""
        error_hashes = {
            r["task_hash"] for r in existing_results
            if r.get("status") == "error"
        }

        return [
            (idx, task) for idx, (task, task_hash) in enumerate(zip(tasks, task_hashes))
            if task_hash in error_hashes
        ]

    def _identify_new_tasks(self, tasks: List[Task], task_hashes: List[str], completed_hashes: set) -> List[tuple]:
        """Identify new tasks that haven't been processed."""
        return [
            (idx, task) for idx, (task, task_hash) in enumerate(zip(tasks, task_hashes))
            if task_hash not in completed_hashes
        ]

    def _process_tasks(self, tasks: List[tuple], output_path: pathlib.Path,
                      check_func: Optional[Callable], max_retries: int,
                      max_workers: int, is_fix: bool = False):
        """Process a list of tasks using thread pool."""
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {
                executor.submit(self._process_single_task, task, idx, output_path,
                              check_func, max_retries, is_fix): idx
                for idx, task in tasks
            }

            try:
                for future in concurrent.futures.as_completed(future_to_task):
                    task_id = future_to_task[future]
                    try:
                        future.result()
                    except Exception as exc:
                        action = "fix" if is_fix else "process"
                        print(f"Task {task_id} generated an exception during {action}: {exc}")
            except KeyboardInterrupt:
                print("Processing interrupted by user.")
                executor.shutdown(wait=False)
                raise

    def _process_single_task(self, task: Task, task_id: int, output_path: pathlib.Path,
                           check_func: Optional[Callable], max_retries: int, is_fix: bool):
        """Process a single task with retry logic."""
        task_hash = self._generate_task_hash(task)

        for attempt in range(max_retries):
            try:
                generated_text = self.generator.generate(task)

                if check_func and not check_func(generated_text):
                    raise ValueError("Check function failed")

                result = self._create_result(task_id, task_hash, task, generated_text, "success", attempt)
                break

            except Exception as e:
                if "sensitive" in str(e).lower():
                    result = self._create_result(
                        task_id, task_hash, task, "Sorry, I can't help with that.",
                        "success", attempt, str(e)
                    )
                    break

                if attempt == max_retries - 1:
                    result = self._create_result(
                        task_id, task_hash, task, "", "error", attempt, str(e)
                    )
                    break

                print(f"Task {task_id} failed: {e}, retrying ({attempt + 1}/{max_retries})")

        with self.results_lock:
            self.results.append(result)
            if not is_fix or result["status"] == "success":
                self._save_result(result, output_path)

    def _create_result(self, task_id: int, task_hash: str, task: Task,
                      generated_text: str, status: str, retries: int,
                      error: Optional[str] = None) -> Dict:
        """Create a result dictionary."""
        messages = task.messages
        if task.prompt:
            use_system = "gemma" not in (task.model_name or "").lower()
            messages = self.generator._prompt_to_messages(task.prompt, use_system)

        result = {
            "task_id": task_id,
            "task_hash": task_hash,
            "model_name": task.model_name,
            "messages": messages,
            "generated_text": generated_text,
            "status": status,
            "retries": retries,
            "timestamp": time.time(),
        }

        if error:
            result["error"] = error

        return result

    def _save_result(self, result: Dict, output_path: pathlib.Path):
        """Save a single result to the output file."""
        with open(output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    def _compile_final_results(self, tasks: List[Task], task_hashes: List[str],
                             existing_results: List[Dict]) -> List[Dict]:
        """Compile final results from all sources."""
        final_results = []

        for idx, task_hash in enumerate(task_hashes):
            result = self._find_best_result(task_hash, existing_results + self.results)
            if result:
                result["task_id"] = idx
                final_results.append(result)
            else:
                final_results.append(None)
                print(f"Task {idx} not found in results.")

        return final_results

    def _find_best_result(self, task_hash: str, all_results: List[Dict]) -> Optional[Dict]:
        """Find the best result for a given task hash (prefer success over error)."""
        success_result = None
        error_result = None

        for result in all_results:
            if result.get("task_hash") == task_hash:
                if result["status"] == "success":
                    success_result = result
                else:
                    error_result = result

        return success_result or error_result

    def _save_final_results(self, results: List[Dict], output_path: pathlib.Path):
        """Save final results to file."""
        with open(output_path, "w", encoding="utf-8") as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")


# Convenience functions for backward compatibility
def llmgen(model_name: str, messages: List[Dict[str, str]], **kwargs) -> str:
    """Legacy function for single LLM generation."""
    generator = LLMGenerator()
    extra_params = {k: v for k, v in kwargs.items()
                   if k not in ["temperature", "max_tokens", "top_p"]}

    task = Task(
        messages=messages,
        model_name=model_name,
        temperature=kwargs.get("temperature", 1.0),
        max_tokens=kwargs.get("max_tokens", 1024),
        top_p=kwargs.get("top_p", 0.95),
        extra_params=extra_params
    )

    return generator.generate(task)


def prompt2msgs(prompt: str, system: bool = True) -> List[Dict[str, str]]:
    """Convert a prompt to messages format."""
    return LLMGenerator._prompt_to_messages(prompt, system)


def llmgen_batch(tasks: Union[List[Dict], str], output_file: str, **kwargs) -> List[Dict]:
    """Legacy function for batch processing."""
    generator = LLMGenerator()
    processor = BatchProcessor(generator)
    return processor.process_batch(tasks, output_file, **kwargs)



if __name__ == "__main__":
    generator = LLMGenerator()

    # Test single generation
    task = Task(prompt="Hi, how are you?", model_name="gpt-4o-2024-11-20")
    try:
        response = generator.generate(task)
        print("Local Response:")
        print(response)
    except Exception as e:
        print(f"Error: {e}")
