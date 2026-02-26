# 🇺🇦 README (UA)

Multi-Agent AI Development System

Система з 4 AI-агентів для автоматизованої розробки Python-проєктів.
Агенти працюють як команда розробників:
	•	Architect — проектує архітектуру
	•	Coder — генерує код
	•	Reviewer — перевіряє якість
	•	Deployer — готує деплой

Система працює локально з open-source моделями (наприклад bigcode/starcoder2-3b) та підтримує Apple Silicon (M1/M2 через MPS) або CPU.

four_agents/
│
├── agents.py          # Класи агентів
├── orchestrator.py    # Логіка взаємодії агентів
├── config.py          # Конфігурації генерації та модель
├── main.py            # Точка входу
└── logs/              # Логи роботи агентів

Як це працює
	1.	Architect створює структуру проєкту
	2.	Coder генерує код
	3.	Reviewer аналізує та повертає статус
	4.	Якщо код відхилено — Coder перегенерує
	5.	Deployer створює Dockerfile, requirements та інструкції

Модель завантажується один раз і використовується всіма агентами.

🇬🇧 README (EN)

Multi-Agent AI Development System

A 4-agent AI system for automated Python project development.
Agents simulate a real software team:
	•	Architect — designs system architecture
	•	Coder — writes code
	•	Reviewer — validates code quality
	•	Deployer — prepares deployment setup

The system runs fully locally using open-source models such as bigcode/starcoder2-3b and supports Apple Silicon (MPS), CUDA, or CPU.
