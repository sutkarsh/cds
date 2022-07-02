"""
Utilities to load configuration files.
Based on github.com/mgharbi/ttools
"""

import os
import yaml

SECTIONS = ["dataset", "model"]


def _merge(default, user):
    """Merges two hierachical dictionaries recursively."""
    if isinstance(user, dict):
        if not isinstance(default, dict):
            print("Warning: Got a dict %s to override a non-dict value %s"
                  % (user, default))
            return user
        for k in user:
            v = user[k]
            if k not in default:
                raise ValueError("Warning: Overriding a parameter not provided"
                                 " in the default config: %s" % k)
            default[k] = _merge(default[k], v)
        return default
    else:
        return user


def parse_config(path=None):
    """Parse one or more .yml configuration files (separated by colons if multiple).

    See config/default.yaml for an example config with the
    possible arguments.

    Args:
        path(str): path to the config file. If none is provided, loads the
        default configuration at root/configs/default.yml
    """

    root = os.path.dirname(os.path.join(os.path.abspath(__file__)))
    print("root", root)
    default_path = os.path.join(root, "config", "default.yml")
    print("def_path", default_path)
    with open(default_path) as fid:
        default = yaml.load(fid, Loader=yaml.FullLoader)

    if path is None:
        conf = default
    else:
        conf = default
        for subpath in path.split(':'):
            with open(subpath) as fid:
                conf_add = yaml.load(fid, Loader=yaml.FullLoader)
            conf = _merge(conf, conf_add)

    for section in conf:
        if section not in SECTIONS:
            raise RuntimeError("Config section '%s' not"
                               " recognized, should be one of"
                               " %s" % (section, SECTIONS))
    return conf
