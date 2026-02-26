import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import time

from four_agents.сonfig import architect_config, coder_config, reviewer_config, deployer_config, model_name, hf_token


# ---------- БАЗОВИЙ АГЕНТ ----------
class BaseAgent:
    def __init__(self, name, system_prompt, gen_config: GenerationConfig):
        self.name = name
        self.system_prompt = system_prompt
        self.gen_config = gen_config

        # Визначення пристрою
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        print(f"[{self.name}] Device: {self.device}")

        # Завантаження токенізатора
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=hf_token
        )

        # Завантаження моделі
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float16 if self.device in ["mps", "cuda"] else torch.float32,
            device_map="auto" if self.device in ["mps", "cuda"] else None,
            token=hf_token
        )

        self.model.eval()

    def run(self, task, context=""):
        print(f"\n[{self.name}] Task received")
        start_time = time.time()

        prompt = (
            f"{self.system_prompt}\n\n"
            f"Context:\n{context}\n\n"
            f"Task:\n{task}\n\n"
            f"Answer:\n"
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        print(f"[{self.name}] Generating...")

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                generation_config=self.gen_config,
                use_cache=True
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
            "You are a senior software architect. "
            "Design project structure. "
            "Do NOT write full code. Return only file tree and responsibilities.",
            architect_config
        )


class CoderAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            "Coder",
            "You are a Python developer. "
            "Write clean production-ready Python code. "
            "Return ONLY code. No explanations.",
            coder_config
        )


class ReviewerAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            "Reviewer",
            "You are a strict senior code reviewer. "
            "Find bugs and bad practices. "
            "Return format: STATUS: approved/rejected\nFIXED_CODE:\n<code>",
            reviewer_config
        )


class DeployerAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            "Deployer",
            "You are a DevOps engineer. "
            "Create requirements.txt, Dockerfile and run instructions.",
            deployer_config
        )
