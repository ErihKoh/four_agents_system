from .orchestrator import Orchestrator
import sys
import os
import torch

# Додаємо корінь проєкту у sys.path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))


def check_device():
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        print("MPS GPU доступний (Apple M1/M2)")
    else:
        print("GPU недоступний, буде використано CPU")


if __name__ == "__main__":
    check_device()
    print("[Main] Старт програми")
    task = "Create a simple Python app with one endpoint that returns current time"
    orchestrator = Orchestrator()
    result = orchestrator.build(task)
    print("[Main] Всі агенти завершили роботу")
