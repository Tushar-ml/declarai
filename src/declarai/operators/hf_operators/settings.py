""" 
Environment level configurations for working with Huggingface LLMs.
"""

import os
from declarai.core.core_settings import DECLARAI_PREFIX

HF_API_KEY: str = os.getenv(
    f"{DECLARAI_PREFIX}_HF_API_KEY", os.getenv("HF_API_KEY", "")
)  # pylint: disable=E1101
"API key for huggingface provider."

