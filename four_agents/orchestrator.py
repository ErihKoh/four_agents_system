from agents import ArchitectAgent, CoderAgent, ReviewerAgent, DeployerAgent
import time


class Orchestrator:
    def __init__(self):
        self.architect = ArchitectAgent()
        self.coder = CoderAgent()
        self.reviewer = ReviewerAgent()
        self.deployer = DeployerAgent()

    def build(self, user_request):
        print("[Orchestrator] Починаємо обробку користувацького запиту")

        # ARCHITECT
        print("\n--- ARCHITECT ---")
        start = time.time()
        spec = self.architect.run(user_request)
        print(spec)
        print(f"[Orchestrator] ARCHITECT завершив за {time.time() - start:.2f} секунд")

        # CODER
        print("\n--- CODER ---")
        start = time.time()
        code = self.coder.run("Implement the architecture", spec)
        print(code)
        print(f"[Orchestrator] CODER завершив за {time.time() - start:.2f} секунд")

        # REVIEWER
        print("\n--- REVIEWER ---")
        start = time.time()
        review = self.reviewer.run("Review this code", code)
        print(review)
        print(f"[Orchestrator] REVIEWER завершив за {time.time() - start:.2f} секунд")

        # Якщо рев’ю показало проблеми — перегенерація
        if "rejected" in review.lower():
            print("\n--- REGENERATING AFTER REVIEW ---")
            start = time.time()
            code = self.coder.run("Fix issues from review", review)
            print(code)
            print(f"[Orchestrator] CODER повторно завершив за {time.time() - start:.2f} секунд")

        # DEPLOYER
        print("\n--- DEPLOYER ---")
        start = time.time()
        deployment = self.deployer.run("Prepare deployment setup", code)
        print(deployment)
        print(f"[Orchestrator] DEPLOYER завершив за {time.time() - start:.2f} секунд")

        print("[Orchestrator] Всі агенти завершили обробку")
        return {"spec": spec, "code": code, "deployment": deployment}