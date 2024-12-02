from node import Node
from model import C4Model
import json
import gzip


def save(model, file_path):
    """
    Saves the model's information, including nodes in node_list, to a file.
    """
    try:
        # data is the information needed to reconstruct a model
        # nodes are all nodes in the model
        # other attributes are things like # rows, # cols and node count
        data = {
            "nodes": {key: node.to_dict() for key, node in model.node_list.items()},
            "other_attributes": model.get_attributes(),
        }


        # Write the data to a file in JSON format
        with open(file_path, "w") as file:
            json.dump(data, file, indent=4)
    except Exception as e:
        print(f"An error occurred while saving the model: {e}")



def save_zip(model, file_path):
    """
    Saves the model's information with gzip compression to reduce file size.
    """
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
    """
    Loads the model's information from a gzip-compressed JSON file.
    """
    try:
        # Open the gzip file for reading
        with gzip.open(file_path, "rt", encoding="utf-8") as file:
            data = json.load(file)

        # Reconstruct the node list and other attributes
        node_list = {key: Node.from_dict(value) for key, value in data["nodes"].items()}
        other_attributes = data["other_attributes"]

        return node_list, other_attributes

    except Exception as e:
        print(f"An error occurred while loading the model: {e}")
        return None, None