import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import os
import time
from dotenv import load_dotenv

load_dotenv()
hf_token = os.getenv("access_token_hf")

if not hf_token:
    raise ValueError("HF token не знайдено у .env файлі під ключем access_token_hf")

model_name = "Qwen/Qwen2.5-Coder-7B-Instruct"

gen_config = GenerationConfig(
    max_new_tokens=256,
    temperature=0.7,
    top_p=0.95,
    top_k=50,
    use_cache=True
)


class BaseAgent:
    def __init__(self, name, system_prompt, model_name=model_name, hf_token=hf_token):
        self.name = name
        self.system_prompt = system_prompt

        # Визначення пристрою: MPS (M1/M2), CUDA, або CPU
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            self.device = "mps"
        else:
            self.device = "cpu"

        print(f"[{self.name}] Використовується пристрій: {self.device}")

        # Завантаження токенізатора та моделі
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_auth_token=hf_token
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float16 if self.device in ["mps"] else torch.float32,
            device_map="auto" if self.device in ["mps"] else None,
            token=hf_token,
            use_cache=True,
        )

    def run(self, task, context=""):
        print(f"[{self.name}] Отримав завдання: {task}")
        start_time = time.time()

        prompt = f"{self.system_prompt}\nContext:\n{context}\nTask:\n{task}\nAnswer:\n"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        print(f"[{self.name}] Починаю генерацію...")
        outputs = self.model.generate(
            **inputs,
            generation_config=GenerationConfig(max_new_tokens=256, use_cache=True)
        )

        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        elapsed = time.time() - start_time
        print(f"[{self.name}] Завершив генерацію за {elapsed:.2f} секунд")
        return result


# 4 агенти з різними ролями
class ArchitectAgent(BaseAgent):
    def __init__(self):
        super().__init__("Architect",
                         "You are a senior software architect. Design project structure. "
                         "Do NOT write full code. Return a clear file list and responsibilities."
                         )


class CoderAgent(BaseAgent):
    def __init__(self):
        super().__init__("Coder",
                         "You are a Python developer. Write clean Python code. "
                         "Do not explain. Return only code."
                         )


class ReviewerAgent(BaseAgent):
    def __init__(self):
        super().__init__("Reviewer",
                         "You are a strict code reviewer. Find bugs and bad patterns. "
                         "Return: STATUS: approved/rejected, FIXED_CODE: <corrected code>"
                         )


class DeployerAgent(BaseAgent):
    def __init__(self):
        super().__init__("Deployer",
                         "You are a DevOps engineer. Create requirements.txt, Dockerfile, run instructions."
                         )
