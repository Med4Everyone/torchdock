"""
Configuration management for TorchDock docking pipeline.

Provides the Config class for loading YAML configuration files and
converting them into attribute-accessible objects, supporting both
default and user-provided configurations.
"""

# Copyright (c) 2026 The TorchDock Authors
#
# This file is part of TorchDock.
# Licensed under the Apache License, Version 2.0. See LICENSE file for details.

import os

import yaml


class Config:
    def __init__(self, config_path=None):
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), "default_config.yaml")

        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)

        self._dict_to_attrs(self.config)

    def _dict_to_attrs(self, d, obj=None):
        """Convert dict to object attributes recursively."""
        if obj is None:
            obj = self
        for key, value in d.items():
            if isinstance(value, dict):
                setattr(obj, key, ConfigDict(value))
            else:
                setattr(obj, key, value)

    def get_config(self):
        return self.config

    def setattr(self, key, value):
        """Set attribute and sync to config dict."""
        if isinstance(value, dict):
            wrapped = ConfigDict(value)
            setattr(self, key, wrapped)
            self.config[key] = value
        else:
            setattr(self, key, value)
            self.config[key] = value

    def __repr__(self):
        lines = [f"  {key}: {value}" for key, value in self.config.items()]
        return "{" + ",".join(lines) + "}"


class ConfigDict:
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, ConfigDict(value))
            else:
                setattr(self, key, value)

    def setattr(self, key, value):
        """Set attribute for ConfigDict."""
        setattr(self, key, value)
