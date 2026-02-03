#!/usr/bin/env python3
"""
Wrapper script to run mini-extra with corrected proxy settings.
Must set environment variables BEFORE any imports.
"""
import os
import sys

# Set proxy environment variables BEFORE any imports
os.environ['NO_PROXY'] = 'localhost,127.0.0.1'
os.environ['no_proxy'] = 'localhost,127.0.0.1'

# Load API configuration
config_file = os.path.expanduser('~/.config/mini-swe-agent/.env')
if os.path.exists(config_file):
    with open(config_file) as f:
        for line in f:
            line = line.strip()
            if '=' in line and not line.startswith('#'):
                key, value = line.split('=', 1)
                value = value.strip("'\"")
                if key == 'api_key':
                    os.environ['OPENAI_API_KEY'] = value
                elif key == 'base_url':
                    os.environ['OPENAI_API_BASE'] = value
                    os.environ['LITELLM_BASE_URL'] = value

# Now import and run
from minisweagent.run.mini_extra import main

if __name__ == "__main__":
    main()
