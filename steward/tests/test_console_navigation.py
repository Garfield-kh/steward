"""Run browser camera input regressions through the Steward Python check entry point."""
from pathlib import Path
import subprocess


def test_console_keyboard_navigation():
    result = subprocess.run(
        ["node", "--test", str(Path(__file__).with_name("console-navigation.test.mjs"))],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
