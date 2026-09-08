"""Verify progress output across the actual subprocess pipe, without training."""
import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest

from tools.run_structure_v1 import run_command


class TerminalBuffer(io.StringIO):
    def isatty(self):
        return True


class ProgressOutputTests(unittest.TestCase):
    def run_child(self, console, fail=False):
        code = (
            "import sys\n"
            "sys.stdout.write('\\rEpoch 1/40 | 1/3 | loss=2.0')\n"
            "sys.stdout.flush()\n"
            "sys.stdout.write('\\rEpoch 1/40 | 2/3 | loss=1.9')\n"
            "sys.stdout.write('\\rEpoch 1/40 | 3/3 | loss=1.8\\n')\n"
            "print('[AMP] scale backoff')\n"
            "sys.stderr.write('diagnostic: '+chr(0x6d4b)+chr(0x8bd5)+'\\n')\n"
        )
        if fail:
            code += "raise RuntimeError('worker failed')\n"
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp)/"run.log"
            with contextlib.redirect_stdout(console):
                if fail:
                    with self.assertRaisesRegex(RuntimeError, "exit code 1"):
                        run_command([sys.executable, "-u", "-c", code], log, "cpu")
                else:
                    run_command([sys.executable, "-u", "-c", code], log, "cpu")
            return log.read_text(encoding="utf-8")

    def test_terminal_keeps_carriage_returns_and_log_saves_progress_snapshots(self):
        console = TerminalBuffer()
        log = self.run_child(console)
        text = console.getvalue()
        self.assertIn("\rEpoch 1/40 | 2/3", text)
        self.assertNotIn("\nEpoch 1/40 | 2/3", text)
        progress = [line for line in log.splitlines() if line.startswith("Epoch 1/40")]
        self.assertEqual(len(progress), 2)
        self.assertIn("1/3", progress[0])
        self.assertIn("3/3", progress[-1])
        self.assertNotIn("\r", log)
        self.assertIn("[AMP] scale backoff", log)
        self.assertIn("diagnostic: 测试", log)

    def test_redirected_console_uses_plain_snapshots(self):
        console = io.StringIO()
        self.run_child(console)
        text = console.getvalue()
        self.assertNotIn("\r", text)
        progress = [line for line in text.splitlines() if line.startswith("Epoch 1/40")]
        self.assertEqual(len(progress), 2)
        self.assertIn("3/3", progress[-1])

    def test_failed_child_preserves_traceback_and_exit_code(self):
        log = self.run_child(TerminalBuffer(), fail=True)
        self.assertIn("Traceback (most recent call last):", log)
        self.assertIn("RuntimeError: worker failed", log)


if __name__ == "__main__":
    unittest.main()
