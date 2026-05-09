#!/usr/bin/env python3
"""Inspect and modify Bloqade Gemini noise model parameters.

Selects the one- or two-zone Gemini Cirq noise model, optionally rewrites all
loss-like parameters with --atom_loss, and prints the resulting public fields.
The same helpers are imported by ghz3.py for model selection and parameter dumps.
"""

import argparse
from dataclasses import asdict, fields, is_dataclass, replace
from typing import Any

from bloqade.cirq_utils import noise


def noise_model_to_items(noise_model: Any):
    if is_dataclass(noise_model):
        return asdict(noise_model).items()
    return vars(noise_model).items()


def dump_noise_parameters(noise_model: Any) -> None:
    for name, value in sorted(noise_model_to_items(noise_model)):
        if not name.startswith("_"):
            print(f"{name}: {value}")


def set_loss_parameters(noise_model: Any, loss: float) -> Any:
    if isinstance(noise_model, dict):
        return {
            key: type(value)(loss)
            if "loss" in str(key).lower() and isinstance(value, (int, float))
            else set_loss_parameters(value, loss)
            for key, value in noise_model.items()
        }
    if isinstance(noise_model, list):
        return [set_loss_parameters(value, loss) for value in noise_model]
    if isinstance(noise_model, tuple):
        return tuple(set_loss_parameters(value, loss) for value in noise_model)

    if is_dataclass(noise_model):
        changes = {}
        for field in fields(noise_model):
            name = field.name
            value = getattr(noise_model, name)
            if name.startswith("_"):
                continue
            if "loss" in name.lower() and isinstance(value, (int, float)):
                changes[name] = type(value)(loss)
            elif is_dataclass(value) or hasattr(value, "__dict__") or isinstance(value, (dict, list, tuple)):
                changes[name] = set_loss_parameters(value, loss)
        return replace(noise_model, **changes)
    elif hasattr(noise_model, "__dict__"):
        items = list(vars(noise_model).items())
    else:
        return noise_model

    for name, value in items:
        if name.startswith("_"):
            continue
        if "loss" in name.lower() and isinstance(value, (int, float)):
            setattr(noise_model, name, type(value)(loss))
        elif is_dataclass(value) or hasattr(value, "__dict__") or isinstance(value, (dict, list, tuple)):
            setattr(noise_model, name, set_loss_parameters(value, loss))
    return noise_model


def select_noise_model(zone_type: int) -> Any:
    if zone_type == 1:
        return noise.GeminiOneZoneNoiseModel()
    if zone_type == 2:
        return noise.GeminiTwoZoneNoiseModel()
    raise ValueError("zone_type must be 1 or 2")


def noise_model_name(zone_type: int) -> str:
    if zone_type == 1:
        return "GeminiOneZoneNoiseModel"
    if zone_type == 2:
        return "GeminiTwoZoneNoiseModel"
    raise ValueError("zone_type must be 1 or 2")


def main() -> None:
    parser = argparse.ArgumentParser()
    prs = parser.add_argument
    prs("--zone_type", type=int, choices=(1, 2), default=1, help="Gemini zone architecture: 1 or 2")
    prs("--atom_loss", type=float, default=0.0, help="if nonzero, set all loss parameters to this value")
    args = parser.parse_args()
    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))

    noise_model = select_noise_model(args.zone_type)
    print(noise_model_name(args.zone_type))

    if args.atom_loss != 0.0:
        noise_model = set_loss_parameters(noise_model, args.atom_loss)

    dump_noise_parameters(noise_model)


if __name__ == "__main__":
    main()
