import shutil
import yaml
from pathlib import Path
from argparse import Namespace
from datetime import datetime


def load_params(path) -> Namespace:
    with open(path, "r") as f:
        params = yaml.safe_load(f)
    return Namespace(**params)


def prepare_directory(base_dir, overwrite=False, copy=True):
    if os.path.exists(base_dir):
        if overwrite:
            shutil.rmtree(base_dir)
            os.makedirs(base_dir)
            return base_dir
        elif copy:
            counter = 0
            while True:
                new_dir = f"{base_dir}{counter}"
                if not os.path.exists(new_dir):
                    os.makedirs(new_dir)
                    return new_dir
                counter += 1
        else:
            return base_dir
    else:
        os.makedirs(base_dir)
        return base_dir
    

def save_config(args: Namespace, base_dir: str | Path, name="config.yaml"):
    config = vars(args).copy()
    config["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for k, v in config.items():
        if isinstance(v, Path):
            config[k] = str(v)

    yaml_path = os.path.join(base_dir, name)
    with open(yaml_path, "w") as f:
        yaml.dump(config, f, sort_keys=False)


def get_params(parser, defaults):
    cli_params = parser.parse_args()
    cli_params = Namespace( **{k: v for k, v in vars(cli_params).items() if v is not None} )

    with open(defaults, "r") as f:
        params = yaml.safe_load(f)
    default_params = Namespace(**params)
    return Namespace( **( vars(default_params) | vars(cli_params)))


def join_params(*all_params):
    arg_dict = dict()
    for x in all_params:
        params = read_params(x)

        arg_dict |= dict(vars(params))
        
    return Namespace(**arg_dict)


def read_params(params: str | dict | Namespace | None):
    if params is None:
        return Namespace()
    if isinstance(params, dict):
        return Namespace(**params)
    if isinstance(params, Namespace):
        return params
    
    if not isinstance(params, str):
        raise TypeError(f"{params = }")
    
    if params.endswith(".yaml") or params.endswith(".yml"):
        with open(params, "r") as f:
            content = yaml.safe_load(f)
            return Namespace(**content)
        
    raise ValueError(f"{params = }")
    