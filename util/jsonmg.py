import logging
import os
import re
from collections import OrderedDict, deque
from typing import Dict, List, Union

import orjson


def extract_json(res: str) -> dict:
    try:
        # Extract json content using regex
        pattern = r"```json\n(.*?)```"
        match = re.search(pattern, res, re.DOTALL)
        if not match:
            raise ValueError("No json content found")
        json_content = match.group(1).strip() + "\n"

        # Parse TOML
        return orjson.loads(json_content)

    except Exception as e:
        print(f"Error extracting json: {e}")
        return None


class JSONManager:
    """
    A class to manage JSON data structure in a file.

    Attributes:
        file_path (str): The file path to the JSON file.
        data (OrderedDict | list): The JSON data structure loaded from the file.
    Methods (public):
        constructor(file_path: str): Initialize the JSONManager with the file path.
        update(new_entry: dict | list, positions: list[dict | str]): Update the JSON data structure at the given positions with the new entry.
    """

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.data = self.load()

    def load(self) -> OrderedDict | List:
        try:
            with open(self.file_path, "rb") as file:
                json_data = orjson.loads(file.read())
                if isinstance(json_data, dict):
                    return OrderedDict(json_data)
                elif isinstance(json_data, list):
                    return json_data
                else:
                    raise ValueError("JSON data must be a dictionary or a list.")
        except orjson.JSONDecodeError:
            return None
        except FileNotFoundError:
            logging.warning(
                f"File '{self.file_path}' not found. An empty JSON data structure will be created.",
            )
            return None

    def dump(self, data: Union[OrderedDict, List] = None) -> None:
        data_to_save = data if data is not None else self.data
        with open(self.file_path, "wb") as file:
            file.write(
                orjson.dumps(
                    data_to_save, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS
                )
            )

    def update(
        self,
        new_entry: Union[Dict, List] = None,
        positions: List[Union[Dict, str]] = None,
    ) -> None:
        """
        Update the JSON data structure at the given positions with the new entry.

        Args:
            new_entry (dict | list): The new part to be added. If new_entry is None, the entry at the given positions will be removed.
            positions (list[dict | str]): The lists of positions(keys) to find in the JSON data structure.
                To locate a dict in a list, the position value should be a subdict to match. (list of list is not allowed)

        Raises:
            TypeError: if the positions are invalid or the data structure is invalid
        """
        if not positions:
            raise TypeError("Positions must be provided for update.")

        current = self.data
        positions = deque(positions)

        while positions:
            pos = positions.popleft()
            if isinstance(current, OrderedDict):
                if pos not in current:
                    raise TypeError(f"Key '{pos}' not found in JSON data.")
                if not positions:
                    if new_entry is None:
                        current.pop(pos)
                    else:
                        current[pos] = new_entry
                else:
                    current = current[pos]
            elif isinstance(current, list):
                if isinstance(pos, dict):
                    matched = False
                    for item in current:
                        if all(item.get(k) == v for k, v in pos.items()):
                            current = item
                            matched = True
                            break
                    if not matched:
                        raise TypeError(
                            f"No matching item found for {pos} in JSON list."
                        )
                elif isinstance(pos, int):
                    if pos < 0 or pos >= len(current):
                        raise TypeError(f"Index '{pos}' out of range for JSON list.")
                    if not positions:
                        if new_entry is None:
                            current.pop(pos)
                        else:
                            current[pos] = new_entry
                    else:
                        current = current[pos]
                else:
                    raise TypeError(
                        "Position must be a dict or integer when JSON data is a list."
                    )
            else:
                raise TypeError("Unsupported JSON data structure.")

    @classmethod
    def update_to(
        cls, file_path: str, new_entry: dict | list, positions: list[dict | str]
    ) -> None:
        """
        Update the JSON data structure at the given positions with the new entry.

        Args:
            file_path (str): The file path to the JSON file.
            new_entry (dict | list): The new part to be added. If new_entry is None, the entry at the given positions will be removed.
            positions (list[dict | str]): The lists of positions(keys) to find in the JSON data structure.
                To locate a dict in a list, the position value should be a subdict to match. (list of list is not allowed)

        Raises:
            TypeError: if the positions are invalid or the data structure is invalid
        """
        manager = cls(file_path)
        manager.update(new_entry, positions)

    @classmethod
    def dump_to(cls, file_path: str, data: dict | list, backup: bool = True) -> None:
        """
        Save the data to the JSON file.

        Args:
            file_path (str): The file path to the JSON file.
            data (dict | list): The data to be saved.
        """
        if os.path.exists(file_path):
            if backup:
                bk_path = f"{file_path}.bak"
                os.rename(file_path, bk_path)
                logging.warning(
                    f"File '{file_path}' already exists, backup the original file at {bk_path} instead"
                )
            else:
                logging.warning(
                    f"File '{file_path}' already exists and will be overwritten.",
                )
        else:
            directory = os.path.dirname(file_path)
            if not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)

        with open(file_path, "wb") as file:
            file.write(
                orjson.dumps(data, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS)
            )
