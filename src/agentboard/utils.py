# -*- coding: utf-8 -*-
# @Time    : 2024/06/27

import codecs
import time
import uuid
import inspect

def read_file(file_path):
    lines = []
    with codecs.open(file_path, "r", "utf-8") as file:
        lines = file.readlines()
    return lines

def save_file(file_path, lines):
    with codecs.open(file_path, "w", "utf-8") as file:
        for line in lines:
            file.write(line + "\n")
    file.close()

def get_current_timestamp():
    timestamp = int(time.time())
    return timestamp

def get_current_timestamp_milli():
    timestamp = round(time.time() * 1000)
    return timestamp

def get_current_datetime():
    import datetime    
    now = datetime.datetime.now()
    datetime = now.strftime('%Y-%m-%d %H:%M:%S')
    return datetime


def get_workflow_id():
    """
        Generate a random UUID as workflow id
    """
    workflow_id = str(uuid.uuid4())
    return workflow_id


def normalize_data_name(name):
    """
        normalize the input name to lower cases and concate with underscore "_"
    """
    name_lower = name.lower()
    words = name_lower.split(" ")
    name_normalize = "_".join(words)
    return name_normalize

### Agent Function Conversion
def function_to_schema(func) -> dict:
    """
        OpenAI Style function calling schema
    """
    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
        type(None): "null",
    }

    try:
        signature = inspect.signature(func)
    except ValueError as e:
        raise ValueError(
            f"Failed to get signature for function {func.__name__}: {str(e)}"
        )

    parameters = {}
    for param in signature.parameters.values():
        try:
            param_type = type_map.get(param.annotation, "string")
        except KeyError as e:
            raise KeyError(
                f"Unknown type annotation {param.annotation} for parameter {param.name}: {str(e)}"
            )
        parameters[param.name] = {"type": param_type}

    required = [
        param.name
        for param in signature.parameters.values()
        if param.default == inspect._empty
    ]

    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": (func.__doc__ or "").strip(),
            "parameters": {
                "type": "object",
                "properties": parameters,
                "required": required,
            },
        },
    }


def function_to_schema_claude(func) -> dict:
    """
        Compatible with Claude's function calling schema
        Ref: 
        https://docs.anthropic.com/en/docs/build-with-claude/tool-use
    """
    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
        type(None): "null",
    }

    try:
        signature = inspect.signature(func)
    except ValueError as e:
        raise ValueError(
            f"Failed to get signature for function {func.__name__}: {str(e)}"
        )

    parameters = {}
    for param in signature.parameters.values():
        try:
            param_type = type_map.get(param.annotation, "string")
        except KeyError as e:
            raise KeyError(
                f"Unknown type annotation {param.annotation} for parameter {param.name}: {str(e)}"
            )
        parameters[param.name] = {"type": param_type}

    required = [
        param.name
        for param in signature.parameters.values()
        if param.default == inspect._empty
    ]

    return {
        "name" : func.__name__, 
        "description": (func.__doc__ or "").strip(),
        "input_schema": {
            "type": "object",
            "properties": parameters,
            "required": required
        }
    }
