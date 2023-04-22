import json
import pandas as pd
from tree import Node
from typing import Dict, Tuple
from pathlib import Path

def save_object(x, file_name):
    with open(file_name, "wb") as file:
        pd.to_pickle(x, file)


def load_object(file_name):
    x = pd.read_pickle(file_name)
    return x


def save_json(data, json_path):
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def load_json(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    return data


def save_checkpoint(out_dir: str, objective_node: Node, embedded_data: Dict):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    objective_node_path = out_dir / "objective_node.pkl"
    embedded_data_path = out_dir / "embedded_data.json"
    save_object(objective_node, objective_node_path)
    save_json(embedded_data, embedded_data_path)


def maybe_load_checkpoint(in_dir: str) -> Tuple[Node, Dict]:
    objective_node_path = Path(in_dir) / "objective_node.pkl"
    embedded_data_path = Path(in_dir) / "embedded_data.json"
    
    objective_node, embedded_data = None, None
    if objective_node_path.is_file():
        objective_node = load_object(objective_node_path)
    if embedded_data_path.is_file():
        embedded_data = load_json(embedded_data_path)
    return objective_node, embedded_data




