import pathlib
import re

import tomli

from util.jsonmg import JSONManager

PROJECT_DIR = pathlib.Path(__file__).resolve().parents[1]

def load_taxonomy(taxonomy_path: pathlib.Path = PROJECT_DIR / "taxonomy.toml") -> dict:
    """Load the taxonomy from a TOML file.

    Args:
        taxonomy_path (pathlib.Path): Path to the taxonomy TOML file.
            Defaults to 'taxonomy.toml' in the project directory.
    Returns:
        dict: The loaded taxonomy.
    """
    with open(taxonomy_path, "rb") as f:
        taxonomy = tomli.load(f)
    return taxonomy

TAXONOMY = load_taxonomy()
ISO2LANG = {lang["iso"]: lang["name"] for lang in TAXONOMY.get("languages", [])}
LANG2ISO = {lang["name"]: lang["iso"] for lang in TAXONOMY.get("languages", [])}


def create_rejection_evaluator(keywords):
    # 1. Escape special regex characters in keywords and join with '|' (OR operator)
    escaped_keywords = [re.escape(keyword) for keyword in keywords]
    regex_pattern = r"(" + "|".join(escaped_keywords) + r")"
    # 2. Compile the regex for efficiency (optional but recommended)
    rejection_regex = re.compile(
        regex_pattern, re.IGNORECASE
    )  # re.IGNORECASE for case-insensitive matching

    # 3. Define the evaluator function
    def is_rejection(text):
        """
        Evaluates if a given string is a rejection based on the compiled regex.

        Args:
            text: The string to evaluate.

        Returns:
            True if the string contains a rejection keyword, False otherwise.
        """
        return bool(
            rejection_regex.search(text)
        )  # re.search() returns None if no match, otherwise a match object. bool(None) is False.

    return is_rejection


jm = JSONManager(str(PROJECT_DIR / "linguasafe" / "kw.json"))
multi_keywords = jm.data
rejection_keywords = []
for d in multi_keywords:
    rejection_keywords.extend([value for _, value in d.items()])

reject_evaluator = create_rejection_evaluator(rejection_keywords)
