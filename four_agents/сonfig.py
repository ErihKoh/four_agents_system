from dotenv import load_dotenv
from transformers import GenerationConfig
import os

load_dotenv()
hf_token = os.getenv("access_token_hf")

thinking_model_name = "Nanbeige/Nanbeige4-3B-Thinking"
tool_model_name = "Nanbeige/ToolMind-Web-3B"

# ---------- КОНФІГУРАЦІЇ ГЕНЕРАЦІЇ ----------

architect_config = GenerationConfig(
    max_new_tokens=256,
    temperature=0.3,
    top_p=0.9,
    do_sample=True
)

coder_config = GenerationConfig(
    max_new_tokens=400,
    do_sample=False   # детермінований код
)

reviewer_config = GenerationConfig(
    max_new_tokens=300,
    do_sample=False   # стабільний аналіз
)

deployer_config = GenerationConfig(
    max_new_tokens=300,
    temperature=0.2,
    top_p=0.9,
    do_sample=True
)
