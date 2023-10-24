from typing import Dict
import logging
import json

configs: Dict[str, any] = {}


def load_json_config(filepath: str = None) -> json:
    global configs
    if not filepath:
        logging.warning("配置文件路径为空")
        return None

    if filepath not in configs:
        with open(filepath) as config_file:
            json_data = json.load(config_file)
        configs[filepath] = json_data

    return configs[filepath]
