"""EVAL FIXTURE — clean look-alike: yaml.safe_load."""
import yaml


def parse_config(text):
    # SAFE: safe_load, not yaml.load
    return yaml.safe_load(text)
