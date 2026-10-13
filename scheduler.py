import importlib
from datetime import datetime
from pathlib import Path

AGENTS_DIR = Path("agents")
OUTPUT_DIR = Path("output")


def load_agents():
    agents = []

    for path in sorted (AGENTS_DIR.glob("*.py")):
        if path.name.startswith("_"):
            continue

        module = importlib.import_module(f"agents.{path.stem}")

        if hasattr(module, "NAME") and hasattr(module, "run"):
            agents.append(module)
        else:
            print(f"Skipping {path.name}: missing NAME or run function")
    return agents

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    for agent in load_agents():
        print(f"[run] {agent.NAME}")

        try:
            output = agent.run()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = OUTPUT_DIR / f"{agent.NAME}_{timestamp}.md"

            out_path.write_text(output)
            print(f"[ok] {agent.NAME} -> {out_path}")
        except Exception as e:
            print(f"[error] {agent.NAME}: {e}")

if __name__ == "__main__":
    main()