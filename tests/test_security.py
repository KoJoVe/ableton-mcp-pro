import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock
import re


ROOT = Path(__file__).resolve().parents[1]


class FakeMCP:
    def __init__(self, *args, **kwargs):
        pass

    def tool(self):
        return lambda function: function

    def run(self):
        pass


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_server(read_only=None, allow_destructive=None):
    fastmcp = types.ModuleType("mcp.server.fastmcp")
    fastmcp.FastMCP = FakeMCP
    fastmcp.Context = type("Context", (), {})
    server_package = types.ModuleType("mcp.server")
    mcp_package = types.ModuleType("mcp")
    environment = {}
    if read_only is not None:
        environment["ABLETON_MCP_READ_ONLY"] = read_only
    if allow_destructive is not None:
        environment["ABLETON_MCP_ALLOW_DESTRUCTIVE"] = allow_destructive
    with mock.patch.dict(
        os.environ,
        environment,
        clear=True,
    ), mock.patch.dict(sys.modules, {
        "mcp": mcp_package,
        "mcp.server": server_package,
        "mcp.server.fastmcp": fastmcp,
    }):
        return load_module(
            "server_under_test_" + os.urandom(4).hex(),
            ROOT / "MCP_Server" / "server.py",
        )


def load_remote_script(read_only=None, allow_destructive=None):
    framework = types.ModuleType("_Framework")
    control_surface_module = types.ModuleType("_Framework.ControlSurface")
    control_surface_module.ControlSurface = type("ControlSurface", (), {})
    environment = {}
    if read_only is not None:
        environment["ABLETON_MCP_READ_ONLY"] = read_only
    if allow_destructive is not None:
        environment["ABLETON_MCP_ALLOW_DESTRUCTIVE"] = allow_destructive
    with mock.patch.dict(os.environ, environment, clear=True), mock.patch.dict(
        sys.modules,
        {
            "_Framework": framework,
            "_Framework.ControlSurface": control_surface_module,
        },
    ):
        return load_module(
            "remote_under_test_" + os.urandom(4).hex(),
            ROOT / "AbletonMCP_Remote_Script" / "__init__.py",
        )


class ServerSafetyTests(unittest.TestCase):
    def test_safe_defaults(self):
        server = load_server()
        self.assertTrue(server.READ_ONLY_MODE)
        self.assertFalse(server.ALLOW_DESTRUCTIVE)
        self.assertTrue(server.REDACT_FILE_PATHS)

    def test_read_only_blocks_changes_before_connecting(self):
        server = load_server()
        connection = server.AbletonConnection("127.0.0.1", 9877)
        with self.assertRaises(PermissionError):
            connection.send_command("set_tempo", {"tempo": 120})
        self.assertIsNone(connection.sock)

    def test_destructive_commands_need_explicit_confirmation(self):
        server = load_server(read_only="0", allow_destructive="1")
        connection = server.AbletonConnection("127.0.0.1", 9877)
        with self.assertRaises(PermissionError):
            connection.send_command("delete_track", {"track_index": 0})
        self.assertIsNone(connection.sock)

    def test_every_destructive_tool_exposes_confirmation_parameter(self):
        server = load_server()
        for command in server.DESTRUCTIVE_COMMANDS:
            signature = inspect.signature(getattr(server, command))
            self.assertIn("confirm_destructive", signature.parameters, command)
            self.assertIs(signature.parameters["confirm_destructive"].default, False)

    def test_oversized_command_is_rejected_before_connecting(self):
        server = load_server(read_only="0")
        connection = server.AbletonConnection("127.0.0.1", 9877)
        with self.assertRaises(ValueError):
            connection.send_command("set_track_name", {"name": "x" * 1_100_000})
        self.assertIsNone(connection.sock)

    def test_file_paths_are_redacted_recursively(self):
        server = load_server()
        value = {"clips": [{"file_path": "/work/secret.wav", "name": "kick"}]}
        self.assertEqual(server._redact_sensitive(value)["clips"][0]["file_path"], "<redacted>")


class RemoteScriptSafetyTests(unittest.TestCase):
    def test_remote_script_enforces_read_only_mode(self):
        remote = load_remote_script()
        instance = object.__new__(remote.AbletonMCP)
        instance.song = lambda: object()
        response = instance._process_command({"type": "set_tempo", "params": {"tempo": 120}})
        self.assertEqual(response["status"], "error")
        self.assertIn("read-only", response["message"])

    def test_remote_script_requires_destructive_confirmation(self):
        remote = load_remote_script(read_only="0", allow_destructive="1")
        instance = object.__new__(remote.AbletonMCP)
        instance.song = lambda: object()
        response = instance._process_command({"type": "delete_track", "params": {"track_index": 0}})
        self.assertEqual(response["status"], "error")
        self.assertIn("confirm_destructive", response["message"])

    def test_server_and_remote_policy_lists_match(self):
        server = load_server()
        remote = load_remote_script()
        self.assertEqual(server.MODIFYING_COMMANDS, remote.MODIFYING_COMMANDS)
        self.assertEqual(server.DESTRUCTIVE_COMMANDS, remote.DESTRUCTIVE_COMMANDS)


class ModelSafetyTests(unittest.TestCase):
    def setUp(self):
        self.bridge = load_module(
            "bridge_under_test_" + os.urandom(4).hex(),
            ROOT / "tools" / "midigenai_bridge.py",
        )
        self.bridge._GENERATOR = None

    def test_custom_model_sources_are_disabled_by_default(self):
        with self.assertRaises(ValueError):
            self.bridge._get_generator("attacker/model", "main", "main")

    def test_pinned_files_are_hashed_and_weights_only_is_forced(self):
        observed = {}
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.pt"
            tokenizer = Path(directory) / "tokenizer.json"
            checkpoint.write_bytes(b"safe checkpoint fixture")
            tokenizer.write_bytes(b"{}")
            self.bridge.PINNED_CHECKPOINT_SHA256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            self.bridge.PINNED_TOKENIZER_SHA256 = hashlib.sha256(tokenizer.read_bytes()).hexdigest()

            fake_torch = types.ModuleType("torch")

            def fake_load(*args, **kwargs):
                observed["weights_only"] = kwargs.get("weights_only")
                return {}

            fake_torch.load = fake_load
            fake_midigenai = types.ModuleType("midigenai")
            fake_midigenai.download_v2_files = lambda **kwargs: (checkpoint, tokenizer)

            class FakeGenerator:
                def __init__(self, **kwargs):
                    fake_torch.load("fixture", weights_only=False)

            fake_midigenai.V2Generator = FakeGenerator
            with mock.patch.dict(sys.modules, {"torch": fake_torch, "midigenai": fake_midigenai}):
                self.bridge._get_generator(None, None, None)

        self.assertIs(observed["weights_only"], True)


class DocumentationSafetyTests(unittest.TestCase):
    def test_readme_json_examples_are_valid(self):
        readme = (ROOT / "README.md").read_text()
        for block in re.findall(r"```json\n(.*?)\n```", readme, re.DOTALL):
            json.loads(block)


if __name__ == "__main__":
    unittest.main()
