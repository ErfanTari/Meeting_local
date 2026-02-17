import sys
from pathlib import Path

# Add the project root (one level up from this script) to the path
sys.path.append(str(Path(__file__).parent.parent))

from app.lmstudio_client import translate, summarize_block
from app.lmstudio_client import translate, summarize_block

sample = "The structure behind this is simpler than the industry probably wants you to believe."

print("\n--- TRANSLATION ---")
print(translate(sample, target_lang="English"))

print("\n--- SUMMARY ---")
block = """
[00:00-00:09] The structure behind this is simpler than the industry probably wants you to believe.
[00:03-00:09] Every agent consists of three components: a language model, tools, and memory.
"""
print(summarize_block(block))
