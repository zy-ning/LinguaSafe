import logging
import pathlib
import re
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple, Union

import tomli

from util.llmgen import BatchProcessor, LLMGenerator, Task

from ..util.misc import LANG2ISO, PROJECT_DIR

# PROJECT_DIR = pathlib.Path(__file__).resolve().parents[1]

# Set up logger
logger = logging.getLogger(__name__)


class TATERStage(Enum):
    """Enum for TATER (Transcreate, Estimate, Refine) stages."""
    TRANSCREATE = "transcreate"
    ESTIMATE = "estimate"
    REFINE = "refine"


@dataclass
class TATERConfig:
    """Configuration for TATER translation process."""
    model: str = "gpt-4o-2024-11-20"
    accuracy_threshold: float = 0.8
    max_retries: int = 5
    max_workers: int = 16


@dataclass
class TranslationTask:
    """Represents a single translation task."""
    input_text: str
    src_lang: str
    tgt_lang: str
    note: str = ""
    task_description: str = ""
    translation: Optional[str] = None


@dataclass
class TATERResult:
    """Result from TATER translation process."""
    final_translation: str
    record: Dict
    passed_estimation: bool = False


class PromptManager:
    """Manages loading and formatting of prompts."""

    def __init__(self, prompts_dir: pathlib.Path):
        self.prompts_dir = prompts_dir
        self._prompts = {}
        self._load_prompts()

    def _load_prompts(self):
        """Load all prompt templates."""
        prompt_files = {
            TATERStage.TRANSCREATE: "ter_translate.txt",
            TATERStage.ESTIMATE: "ter_estimate.txt",
            TATERStage.REFINE: "ter_refine.txt",
            "task": "ter_task.txt"
        }

        for key, filename in prompt_files.items():
            try:
                with open(self.prompts_dir / filename, "r", encoding="utf-8") as f:
                    self._prompts[key] = f.read()
            except FileNotFoundError:
                print(f"Warning: Prompt file {filename} not found")
                self._prompts[key] = ""

    def get_prompt(self, stage: Union[TATERStage, str]) -> str:
        """Get prompt template for a given stage."""
        return self._prompts.get(stage, "")

    def format_prompt(self, stage: TATERStage, **kwargs) -> str:
        """Format prompt template with given parameters."""
        template = self.get_prompt(stage)
        return template.format(**kwargs)


class TOMLExtractor:
    """Handles extraction and validation of TOML content from LLM responses."""

    @staticmethod
    def extract_toml(response: str) -> Optional[Dict]:
        """Extract and parse TOML content from response."""
        try:
            pattern = r"```toml\n(.*?)```"
            match = re.search(pattern, response, re.DOTALL)
            if not match:
                raise ValueError("No TOML content found")

            toml_content = match.group(1).strip() + "\n"
            return tomli.loads(toml_content)
        except Exception as e:
            print(f"Error extracting TOML: {e}")
            return None

    @staticmethod
    def check_transcreation(response: str) -> bool:
        """Check if transcreation response is valid."""
        toml_data = TOMLExtractor.extract_toml(response)
        try:
            return toml_data is not None and "Target" in toml_data["transcreate"]
        except (KeyError, TypeError):
            return False

    @staticmethod
    def check_estimation(response: str) -> bool:
        """Check if estimation response is valid."""
        toml_data = TOMLExtractor.extract_toml(response)
        try:
            return toml_data is not None and "task_acc" in toml_data["task_evaluation"]
        except (KeyError, TypeError):
            return False

    @staticmethod
    def check_refinement(response: str) -> bool:
        """Check if refinement response is valid."""
        toml_data = TOMLExtractor.extract_toml(response)
        try:
            return toml_data is not None and "Target" in toml_data["refinement"]
        except (KeyError, TypeError):
            return False


class TATERTranslator:
    """Main class for TATER (Transcreate, Estimate, Refine) translation."""

    def __init__(self, config_path: Optional[pathlib.Path] = None):
        if config_path is None:
            config_path = PROJECT_DIR / "api.toml"

        self.generator = LLMGenerator(config_path)
        self.batch_processor = BatchProcessor(self.generator)
        self.prompt_manager = PromptManager(PROJECT_DIR / "tater" / "prompts")
        self.toml_extractor = TOMLExtractor()

    def translate_single(
        self,
        task: TranslationTask,
        config: TATERConfig = TATERConfig()
    ) -> TATERResult:
        """Translate a single text using TATER process."""
        record = {"input": task.input_text}
        translation = task.translation

        # Step 1: Transcreate (if no translation provided)
        if translation is None:
            translation = self._transcreate_single(task, config, record)

        # Step 2: Estimate
        estimation_result = self._estimate_single(task, translation, config, record)

        # Step 3: Refine (if estimation score is low)
        if estimation_result["task_acc"] <= config.accuracy_threshold:
            translation = self._refine_single(task, translation, estimation_result, config, record)
            passed_estimation = False
        else:
            passed_estimation = True

        return TATERResult(
            final_translation=translation,
            record=record,
            passed_estimation=passed_estimation
        )

    def translate_batch(
        self,
        tasks: List[TranslationTask],
        log_dir: pathlib.Path,
        config: TATERConfig = TATERConfig(),
        check_functions: Optional[Dict[TATERStage, Callable]] = None
    ) -> Tuple[List[str], List[Dict]]:
        """Translate multiple texts using TATER process with batch processing."""

        if check_functions is None:
            check_functions = {
                TATERStage.TRANSCREATE: self.toml_extractor.check_transcreation,
                TATERStage.ESTIMATE: self.toml_extractor.check_estimation,
                TATERStage.REFINE: self.toml_extractor.check_refinement
            }

        records = [{"input": task.input_text} for task in tasks]
        tgt_lang = tasks[0].tgt_lang if tasks else "unknown"
        lang_code = LANG2ISO.get(tgt_lang, tgt_lang.lower())

        # Step 1: Transcreate
        translations = self._transcreate_batch(tasks, records, log_dir, lang_code, config, check_functions)

        # Step 2: Estimate
        pass_ids, ref_ids = self._estimate_batch(
            tasks, translations, records, log_dir, lang_code, config, check_functions
        )

        # Step 3: Refine
        final_translations = self._refine_batch(
            tasks, translations, records, pass_ids, ref_ids, log_dir, lang_code, config, check_functions
        )

        return final_translations, records

    def _transcreate_single(self, task: TranslationTask, config: TATERConfig, record: Dict) -> str:
        """Handle single transcreation."""
        prompt = self.prompt_manager.format_prompt(
            TATERStage.TRANSCREATE,
            INPUT=task.input_text,
            SRC=task.src_lang,
            TGT=task.tgt_lang,
            NOTE=task.note,
            TASK=task.task_description
        )

        llm_task = Task(prompt=prompt, model_name=config.model)
        response = self.generator.generate(llm_task)

        toml_data = self.toml_extractor.extract_toml(response)
        record.update(toml_data or {})

        return toml_data["transcreate"]["Target"] if toml_data else ""

    def _estimate_single(self, task: TranslationTask, translation: str,
                        config: TATERConfig, record: Dict) -> Dict:
        """Handle single estimation."""
        prompt = self.prompt_manager.format_prompt(
            TATERStage.ESTIMATE,
            INPUT=task.input_text,
            SRC=task.src_lang,
            TGT=task.tgt_lang,
            TRANS=translation,
            TASK=task.task_description
        )

        llm_task = Task(prompt=prompt, model_name=config.model)
        response = self.generator.generate(llm_task)

        toml_data = self.toml_extractor.extract_toml(response)
        record.update(toml_data or {})

        return toml_data["task_evaluation"] if toml_data else {"task_acc": 0.0}

    def _refine_single(self, task: TranslationTask, translation: str,
                      estimation: Dict, config: TATERConfig, record: Dict) -> str:
        """Handle single refinement."""
        prompt = self.prompt_manager.format_prompt(
            TATERStage.REFINE,
            INPUT=task.input_text,
            SRC=task.src_lang,
            TGT=task.tgt_lang,
            TRANS=translation,
            TASK=task.task_description,
            EST=str(estimation),
            NOTE=task.note
        )

        llm_task = Task(prompt=prompt, model_name=config.model)
        response = self.generator.generate(llm_task)

        toml_data = self.toml_extractor.extract_toml(response)
        record.update(toml_data or {})

        return toml_data["refinement"]["Target"] if toml_data else translation

    def _transcreate_batch(self, tasks: List[TranslationTask], records: List[Dict],
                          log_dir: pathlib.Path, lang_code: str, config: TATERConfig,
                          check_functions: Dict) -> List[str]:
        """Handle batch transcreation."""
        if any(task.translation for task in tasks):
            return [task.translation or "" for task in tasks]

        logger.info(f"Starting transcreation for {tasks[0].tgt_lang}...")

        batch_tasks = []
        for task in tasks:
            prompt = self.prompt_manager.format_prompt(
                TATERStage.TRANSCREATE,
                INPUT=task.input_text,
                SRC=task.src_lang,
                TGT=task.tgt_lang,
                NOTE=task.note,
                TASK=task.task_description
            )
            batch_tasks.append({"prompt": prompt})

        results = self.batch_processor.process_batch(
            batch_tasks,
            output_file=str(log_dir / f"log_{lang_code}_transcreate.jsonl"),
            default_model=config.model,
            check_func=check_functions[TATERStage.TRANSCREATE],
            max_workers=config.max_workers
        )

        for i, result in enumerate(results):
            if result and result["status"] == "success":
                toml_data = self.toml_extractor.extract_toml(result["generated_text"])
                records[i].update(toml_data or {})

        return [
            records[i].get("transcreate", {}).get("Target", "")
            for i in range(len(tasks))
        ]

    def _estimate_batch(self, tasks: List[TranslationTask], translations: List[str],
                       records: List[Dict], log_dir: pathlib.Path, lang_code: str,
                       config: TATERConfig, check_functions: Dict) -> Tuple[List[int], List[int]]:
        """Handle batch estimation."""
        logger.info(f"Starting estimation for {tasks[0].tgt_lang}...")

        batch_tasks = []
        for task, translation in zip(tasks, translations):
            prompt = self.prompt_manager.format_prompt(
                TATERStage.ESTIMATE,
                INPUT=task.input_text,
                SRC=task.src_lang,
                TGT=task.tgt_lang,
                TRANS=translation,
                TASK=task.task_description
            )
            batch_tasks.append({"prompt": prompt})

        results = self.batch_processor.process_batch(
            batch_tasks,
            output_file=str(log_dir / f"log_{lang_code}_estimate.jsonl"),
            default_model=config.model,
            check_func=check_functions[TATERStage.ESTIMATE],
            max_workers=config.max_workers
        )

        for i, result in enumerate(results):
            if result and result["status"] == "success":
                toml_data = self.toml_extractor.extract_toml(result["generated_text"])
                records[i].update(toml_data or {})

        pass_ids = []
        ref_ids = []

        for i, record in enumerate(records):
            task_acc = record.get("task_evaluation", {}).get("task_acc", 0.0)
            if task_acc > config.accuracy_threshold:
                pass_ids.append(i)
            else:
                ref_ids.append(i)

        return pass_ids, ref_ids

    def _refine_batch(self, tasks: List[TranslationTask], translations: List[str],
                     records: List[Dict], pass_ids: List[int], ref_ids: List[int],
                     log_dir: pathlib.Path, lang_code: str, config: TATERConfig,
                     check_functions: Dict) -> List[str]:
        """Handle batch refinement."""
        if not ref_ids:
            return translations

        logger.info(f"Starting refinement for {tasks[0].tgt_lang}...")

        batch_tasks = []
        for ref_id in ref_ids:
            task = tasks[ref_id]
            translation = translations[ref_id]
            estimation = records[ref_id].get("task_evaluation", {})

            prompt = self.prompt_manager.format_prompt(
                TATERStage.REFINE,
                INPUT=task.input_text,
                SRC=task.src_lang,
                TGT=task.tgt_lang,
                TRANS=translation,
                TASK=task.task_description,
                EST=str(estimation),
                NOTE=task.note
            )
            batch_tasks.append({"prompt": prompt})

        # Use higher retries for certain models
        max_retries = 10 if config.model == "qwen" else config.max_retries

        results = self.batch_processor.process_batch(
            batch_tasks,
            output_file=str(log_dir / f"log_{lang_code}_refine.jsonl"),
            default_model=config.model,
            check_func=check_functions[TATERStage.REFINE],
            max_retries=max_retries,
            max_workers=config.max_workers
        )

        # Update records with refinement results
        for i, (ref_id, result) in enumerate(zip(ref_ids, results)):
            if result and result["status"] == "success":
                toml_data = self.toml_extractor.extract_toml(result["generated_text"])
                if toml_data:
                    records[ref_id].update(toml_data)

        # Create final translations list
        final_translations = translations.copy()
        for ref_id in ref_ids:
            refinement_data = records[ref_id].get("refinement", {})
            if "Target" in refinement_data:
                final_translations[ref_id] = refinement_data["Target"]

        return final_translations


# Convenience functions for backward compatibility
def extract_toml(trans_res: str) -> Dict:
    """Legacy function for TOML extraction."""
    return TOMLExtractor.extract_toml(trans_res)


def check_trans(s: str) -> bool:
    """Legacy function for transcreation validation."""
    return TOMLExtractor.check_transcreation(s)


def check_est(s: str) -> bool:
    """Legacy function for estimation validation."""
    return TOMLExtractor.check_estimation(s)


def check_ref(s: str) -> bool:
    """Legacy function for refinement validation."""
    return TOMLExtractor.check_refinement(s)


def tater(
    input_str: str,
    tgt_lang: str,
    src_lang: str,
    translation: str = None,
    note: str = "",
    task: str = None,
    model: str = "gpt-4o-2024-11-20",
) -> Tuple[str, Dict]:
    """Legacy function for single TATER translation."""
    translator = TATERTranslator()

    translation_task = TranslationTask(
        input_text=input_str,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        note=note,
        task_description=task or "",
        translation=translation
    )

    config = TATERConfig(model=model)
    result = translator.translate_single(translation_task, config)

    return result.final_translation, result.record


def tater_batch(
    input_tasks: List[str],
    tgt_lang: str,
    src_lang: str,
    log_dir: str,
    overwrite_check_trans: Callable = None,
    overwrite_check_est: Callable = None,
    overwrite_check_ref: Callable = None,
    translations: List[str] = None,
    note: str = "",
    task: str = None,
    model: str = "gpt-4o-2024-11-20",
) -> Tuple[List[str], List[Dict]]:
    """Legacy function for batch TATER translation."""
    translator = TATERTranslator()

    tasks = []
    for i, input_text in enumerate(input_tasks):
        trans = translations[i] if translations else None
        tasks.append(TranslationTask(
            input_text=input_text,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            note=note,
            task_description=task or "",
            translation=trans
        ))

    check_functions = {}
    if overwrite_check_trans:
        check_functions[TATERStage.TRANSCREATE] = overwrite_check_trans
    if overwrite_check_est:
        check_functions[TATERStage.ESTIMATE] = overwrite_check_est
    if overwrite_check_ref:
        check_functions[TATERStage.REFINE] = overwrite_check_ref

    config = TATERConfig(model=model)

    return translator.translate_batch(
        tasks,
        pathlib.Path(log_dir),
        config,
        check_functions if check_functions else None
    )

