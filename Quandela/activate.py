#!/usr/bin/env python3

import os
import perceval as pcvl
from perceval.runtime import RemoteConfig

print(f"Perceval version: {pcvl.__version__}")

# Read token from environment variable
token = os.environ.get("MY_QUANDELA_TOKEN")
if not token:
    raise ValueError("PERCEVAL_TOKEN environment variable is not set.")

# Create config instance
rc = RemoteConfig()
rc.set_token(token)
rc.save()

print("✅ Token saved successfully to ~/.perceval/config.json")
