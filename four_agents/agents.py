import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import time

from four_agents.сonfig import architect_config, coder_config, reviewer_config, deployer_config, hf_token, thinking_model_name


# ---------- БАЗОВИЙ АГЕНТ ----------
class BaseAgent:
    def __init__(self, name, system_prompt, gen_config: GenerationConfig, model_name: str):
        self.name = name
        self.system_prompt = system_prompt
        self.gen_config = gen_config

        # Визначення пристрою
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            self.device = "mps"
        else:
            self.device = "cpu"

        print(f"[{self.name}] Device: {self.device}")

        # Токенізатор
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
        )

        # Модель
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto" if self.device == "mps" else None,
            torch_dtype="auto",  # автоматично обирає float16/float32
            token=hf_token,
        )
        self.model.eval()

    def run(self, task, context=""):
        print(f"\n[{self.name}] Task received: {task}")
        start_time = time.time()

        prompt = f"{self.system_prompt}\nContext:\n{context}\nTask:\n{task}\nAnswer:\n"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        print(f"[{self.name}] Generating...")
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.gen_config.max_new_tokens,
                temperature=self.gen_config.temperature,
                top_p=self.gen_config.top_p,
                do_sample=self.gen_config.do_sample,
                use_cache=True,
                eos_token_id=self.tokenizer.eos_token_id
            )

        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        elapsed = time.time() - start_time
        print(f"[{self.name}] Done in {elapsed:.2f}s")
        return result


# ---------- АГЕНТИ ----------
class ArchitectAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            "Architect",
            "You are a senior software architect. Design project structure. Do NOT write full code.",
            architect_config,
            thinking_model_name
        )


class CoderAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            "Coder",
            "You are a Python developer. Write clean production-ready Python code. Return ONLY code.",
            coder_config,
            thinking_model_name
        )


class ReviewerAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            "Reviewer",
            "You are a strict senior code reviewer. Find bugs and bad practices. Return: STATUS: "
            "approved/rejected\nFIXED_CODE:\n<code>",
            reviewer_config,
            thinking_model_name
        )


class DeployerAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            "Deployer",
            "You are a DevOps engineer. Create requirements.txt, Dockerfile and run instructions.",
            deployer_config,
            thinking_model_name
        )
