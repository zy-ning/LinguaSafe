import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd


def _eval_module():
    repo_root = Path(__file__).resolve().parents[1]
    root_str = str(repo_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return importlib.import_module("linguasafe.eval")


def _evaluator(eval_mod):
    evaluator = eval_mod.LinguaSafeEvaluator.__new__(eval_mod.LinguaSafeEvaluator)
    evaluator.args = SimpleNamespace(
        model="gpt-4o-2024-11-20",
        lang="all",
        generalization=False,
        jailbreak=None,
        evaluator="gpt-4o-2024-11-20",
        assitSLM="internlm3-8b-instruct",
        limit=64,
    )
    return evaluator


def test_parse_arguments_supports_jailbreak(monkeypatch):
    eval_mod = _eval_module()
    monkeypatch.setattr(sys, "argv", ["prog", "--jailbreak", "DAN"])
    args = eval_mod.parse_arguments()
    assert args.jailbreak == "DAN"


def test_rejection_uses_full_first_line(monkeypatch):
    eval_mod = _eval_module()
    evaluator = _evaluator(eval_mod)

    monkeypatch.setattr(eval_mod, "reject_evaluator", lambda text: "laterkey" in text)

    df = pd.DataFrame(
        [
            {
                "prompt": "p",
                "lang": "en",
                "subtype": "toxicity",
                "level": 1,
                "specific": True,
                "rejected": False,
            }
        ],
        index=pd.Index([42]),
    )

    record = {
        "prompt": "p",
        "lang": "en",
        "subtype": "toxicity",
        "level": 1,
        "specific": True,
    }
    response = {"generated_text": "prefix text laterkey\nrest"}
    evaluation_counts = evaluator._initialize_evaluation_counts(["en"], ["toxicity"])

    evaluator._evaluate_individual_response(
        record=record,
        response=response,
        rec_idx=0,
        df=df,
        record_to_df_idx={0: 42},
        evaluation_counts=evaluation_counts,
    )

    assert record["reject"] is True
    assert bool(df.loc[42, "rejected"]) is True


def test_mcq_none_generated_text_matches_original_behavior():
    eval_mod = _eval_module()
    evaluator = _evaluator(eval_mod)

    evaluation_counts = evaluator._initialize_evaluation_counts(["en"], ["toxicity"])
    record_list = [
        {"lang": "en", "subtype": "toxicity", "level": 3, "specific": True},
    ]

    evaluator._evaluate_mcq_response(record_list, {"generated_text": None}, evaluation_counts)

    ec = evaluation_counts["en"]["toxicity"]
    assert ec.mcq_cm[0, 0] == 1


def test_save_results_keeps_original_scores_schema(tmp_path, monkeypatch):
    eval_mod = _eval_module()
    evaluator = _evaluator(eval_mod)

    monkeypatch.setattr(eval_mod, "PROJECT_DIR", tmp_path)

    df = pd.DataFrame([{"a": 1, "rejected": False}])
    scores = {"en": {"tox": {"unsafe rate": 0.1}}}
    ge_scores = {"en": {"tox": {"unsafe rate": 0.2}}}

    _, scores_path = evaluator._save_results(df, scores, ge_scores, tag="")

    payload = json.loads(scores_path.read_text())
    assert isinstance(payload, list)
    assert payload[0] == scores
    assert payload[1] == ge_scores


def test_jailbreak_transforms_prompt(monkeypatch):
    eval_mod = _eval_module()
    evaluator = _evaluator(eval_mod)
    evaluator.args.jailbreak = "DAN"

    row = pd.Series(
        {
            "id": float("nan"),
            "prompt": "How to do X?",
            "lang": "en",
            "subtype": "toxicity",
            "specific": True,
            "level": 1,
        }
    )
    df = pd.DataFrame([row])

    task, _, should_continue = evaluator._process_row(
        row=row,
        row_idx=0,
        mcq_task=[],
        mcq_record=[],
        use_mcq=True,
        tasks=[],
        records=[],
        df=df,
        record_to_df_idx={},
    )

    assert should_continue is False
    assert task["prompt"] != "How to do X?"
    assert "How to do X?" in task["prompt"]
