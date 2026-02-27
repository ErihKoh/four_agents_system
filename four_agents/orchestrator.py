from .agents import ArchitectAgent, CoderAgent, ReviewerAgent, DeployerAgent
import time
from pathlib import Path


class Orchestrator:
    def __init__(self, log_dir="logs"):
        self.architect = ArchitectAgent()
        self.coder = CoderAgent()
        self.reviewer = ReviewerAgent()
        self.deployer = DeployerAgent()
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True, parents=True)

    def _run_agent(self, agent, task, context="", filename=None):
        start = time.time()
        result = agent.run(task, context)
        elapsed = time.time() - start
        print(f"[Orchestrator] {agent.name} завершив за {elapsed:.2f} секунд")

        # Збереження логів
        if filename:
            filepath = self.log_dir / filename
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(result)
            print(f"[Orchestrator] Результат {agent.name} збережено у {filepath}")

        return result

    def build(self, user_request):
        print("[Orchestrator] Починаємо обробку користувацького запиту")

        # ARCHITECT
        print("\n--- ARCHITECT ---")
        spec = self._run_agent(self.architect, user_request, filename="architect_spec.txt")

        # CODER
        print("\n--- CODER ---")
        code = self._run_agent(self.coder, "Implement the architecture", spec, filename="code.py")

        # REVIEWER
        print("\n--- REVIEWER ---")
        review = self._run_agent(self.reviewer, "Review this code", code, filename="review.txt")

        # Якщо рев’ю показало проблеми — перегенерація
        if "rejected" in review.lower():
            print("\n--- REGENERATING AFTER REVIEW ---")
            code = self._run_agent(self.coder, "Fix issues from review", review, filename="code_fixed.py")

        # DEPLOYER
        print("\n--- DEPLOYER ---")
        deployment = self._run_agent(self.deployer, "Prepare deployment setup", code, filename="deployment.txt")

        print("[Orchestrator] Всі агенти завершили обробку")
        return {"spec": spec, "code": code, "deployment": deployment}
