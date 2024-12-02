from node import Node
from model import C4Model
import json
import gzip

def save_zip(model, file_path):
    '''
        Converts the models data into JSON format then
        saves the model information with gzip
    '''
    try:
        data = {
            "nodes": {key: node.to_dict() for key, node in model.node_list.items()},
            "other_attributes": model.get_attributes(),
        }

        with gzip.open(file_path, "wt", encoding="utf-8") as file:
            json.dump(data, file, separators=(",", ":"), ensure_ascii=False)  # Compact JSON

    except Exception as e:
        print(f"An error occurred while saving the model: {e}")


def load(file_path):
    '''
    Loads the model information from a gzip-compressed JSON file.
    '''
    try:
        with gzip.open(file_path, "rt", encoding="utf-8") as file:
            data = json.load(file)

        node_list = {key: Node.from_dict(value) for key, value in data["nodes"].items()}
        other_attributes = data["other_attributes"]

        return node_list, other_attributes

    except Exception as e:
        print(f"An error occurred while loading the model: {e}")
        return None, None