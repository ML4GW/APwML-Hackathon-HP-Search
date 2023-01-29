import argparse
import inspect
import importlib
import re
import shutil
from functools import partial
from typing import Callable, Dict

import ray
from typeo.scriptify import make_parser


def get_search_space(search_space: str) -> Dict[str, Callable]:
    """Import a search space dictionary from a python file

    File is expected to functioning standalone pythong
    script which defines a `search_space` dictionary
    that can be used by `ray`, e.g.
    ```python
    # search_space.py
    import ray

    search_space = {
        "learning_rate": ray.tune.loguniform(1e-5, 1e-3),
        "hidden_size": ray.tune.choice([64, 128, 256])
    }
    ```
    """

    locals_dict = {}
    try:
        with open(search_space, "r") as f:
            exec(f.read(), {}, locals_dict)
    except FileNotFoundError as e:
        if search_space in str(e):
            raise ValueError(f"File {search_space} does not exist")
        raise

    try:
        return locals_dict["search_space"]
    except KeyError:
        raise ValueError(f"'{search_space}' has no variable 'search_space'.")


def get_train_fn(executable: str) -> Callable:
    try:
        library, fn = executable.split(":")
    except ValueError:
        executable_path = shutil.which(executable)
        if executable_path is None:
            raise ValueError(f"{executable}: command not found")

        import_re = re.compile(
            "(?m)^from (?P<lib>[a-zA-Z0-9_.]+) import (?P<fn>[a-zA-Z0-9_]+)$"
        )
        with open(executable_path, "r") as f:
            match = import_re.search(f.read())
            if match is None:
                raise ValueError(
                    "Could not find library to import in "
                    "executable at path {}".format(executable_path)
                )
            library = match.group("lib")
            fn = match.group("fn")

    module = importlib.import_module(library)
    return getattr(module, fn)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "executable",
        type=str,
        help=(
            "Python executable to run as worker. Can either "
            "be the name of an installed console script, or "
            "be passed in the form `<filename>:<function>`, where "
            "`<function>` will be imported from `<filename>`."
        )
    )
    parser.add_argument(
        "search-space",
        type=str,
        help="Path to file containing search space"
    )
    parser.add_argument(
        "--address",
        type=str,
        required=True,
        help="Parameter server API endpoint"
    )
    args, remainder = parser.parse_known_args()
    ray.init(args.address)

    # find out the names of the hyperparameters we'll
    # be searching over and remove them from the
    # signature of the training function
    search_space = get_search_space(args.search_space)
    train_func = get_train_fn(args.executable)
    parameters = inspect.signature(train_func).parameters
    extra_params = [p for p in search_space if p not in parameters]
    if extra_params:
        raise ValueError(
            "Search space contained extra arguments {}".format(
                ", ".join(extra_params)
            )
        )
    new_params = [v for k, v in parameters.items() if k not in search_space]

    # reassign the signature of the search function
    # to automatically build a new parser for all
    # the non-searched args, then parse them from
    # whatever was leftover from the first parser
    train_func.__signature__ = inspect.Signature(parameters=new_params)
    train_parser = argparse.ArgumentParser(add_help=False)
    train_parser = make_parser(train_func, parser)
    train_args = train_parser.parse_args(remainder)

    # create a partial function that gets called
    # on each run of the tune trial
    train_args = dict(vars(train_args))
    train_partial = partial(train_func, **train_args)

    def objective(config):
        return train_partial(**config)

    tuner = ray.tune.Tuner(objective, param_space=search_space)
    tuner.tune()


if __name__ == "__main__":
    main()
