#!/usr/bin/env python3
"""
FlyProject: Automated UI & Dashboard Test Suite
================================================
Validates DOM elements, telemetry widgets, interactive canvas simulation logic,
and local HTTP server integration for the Drosophila Swarm Mesh Dashboard.

Usage:
  python3 tests/test_ui.py
"""

import os
import sys
import time
import socket
import threading
import unittest
import urllib.request
from html.parser import HTMLParser

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


class DOMElementExtractor(HTMLParser):
    """Parses HTML5 and collects element tags, IDs, classes, and text."""
    def __init__(self):
        super().__init__()
        self.ids = set()
        self.tags = set()
        self.buttons = []
        self.canvas_ids = []
        self.inputs = []

    def handle_starttag(self, tag, attrs):
        self.tags.add(tag)
        attr_dict = dict(attrs)
        if "id" in attr_dict:
            self.ids.add(attr_dict["id"])
        if tag == "canvas" and "id" in attr_dict:
            self.canvas_ids.append(attr_dict["id"])
        if tag == "button":
            self.buttons.append(attr_dict.get("id", ""))
        if tag == "input":
            self.inputs.append(attr_dict.get("id", ""))


class TestUIDashboardStatic(unittest.TestCase):
    """Verifies dashboard file integrity, HTML5 structure, and DOM element definitions."""

    def setUp(self):
        self.dashboard_path = os.path.join(PROJECT_ROOT, "ui", "dashboard.html")
        self.assertTrue(os.path.exists(self.dashboard_path), f"Dashboard missing at {self.dashboard_path}")
        with open(self.dashboard_path, "r", encoding="utf-8") as f:
            self.html_content = f.read()
        self.parser = DOMElementExtractor()
        self.parser.feed(self.html_content)

    def test_file_size_and_encoding(self):
        self.assertGreater(len(self.html_content), 5000, "Dashboard HTML content is unexpectedly small")
        self.assertIn("<!DOCTYPE html>", self.html_content)
        self.assertIn("<html lang=\"en\">", self.html_content)

    def test_interactive_canvas_elements(self):
        # Canvas must be defined for 2D swarm simulation
        self.assertIn("swarmCanvas", self.parser.canvas_ids, "Mandatory #swarmCanvas missing")
        self.assertIn("swarmCanvas", self.parser.ids)

    def test_control_buttons(self):
        # Essential control buttons
        self.assertIn("btnPlay", self.parser.ids, "Missing #btnPlay trigger button")
        self.assertIn("btnReset", self.parser.ids, "Missing #btnReset reset button")

    def test_telemetry_and_hud_labels(self):
        # Dynamic telemetry display labels
        expected_labels = ["cascadeTimer", "alertedCount", "waveSpeed", "meshStatus"]
        for lbl in expected_labels:
            self.assertIn(lbl, self.parser.ids, f"HUD status element #{lbl} missing")

    def test_modality_toggles(self):
        # Multi-modal channel toggles
        expected_toggles = ["toggleVisual", "toggleAcoustic", "togglePheromone", "toggleFog", "toggleMortality"]
        for tog in expected_toggles:
            self.assertIn(tog, self.parser.ids, f"Control toggle #{tog} missing")

    def test_all_six_milestones_present(self):
        # Comparative matrix must document Milestones 1 to 6
        for m in range(1, 7):
            milestone_header = f"Milestone {m}"
            self.assertIn(milestone_header, self.html_content, f"Milestone {m} card missing from dashboard matrix")

    def test_simulation_script_logic(self):
        # Verify core JavaScript functions exist
        expected_js_symbols = [
            "initSwarm",
            "updateSimulation",
            "drawSwarm",
            "__RUN_AUTOMATED_UI_TESTS__",
            "requestAnimationFrame"
        ]
        for sym in expected_js_symbols:
            self.assertIn(sym, self.html_content, f"JavaScript symbol {sym} missing from client script")


class TestUIServerEndpoint(unittest.TestCase):
    """Verifies that the dashboard serves correctly over HTTP with valid headers and static assets."""

    @classmethod
    def setUpClass(cls):
        import http.server
        import functools

        cls.server_available = False
        try:
            # Test if loopback socket is permitted
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('127.0.0.1', 0))
                cls.port = s.getsockname()[1]
                handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=PROJECT_ROOT)
                cls.httpd = http.server.HTTPServer(('127.0.0.1', cls.port), handler)
                cls.server_thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
                cls.server_thread.start()
                time.sleep(0.05)
                # Verify connection can be opened
                with socket.create_connection(('127.0.0.1', cls.port), timeout=1.0):
                    pass
                cls.server_available = True
        except (PermissionError, OSError):
            cls.server_available = False

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, 'server_available', False) and hasattr(cls, 'httpd'):
            cls.httpd.shutdown()
            cls.httpd.server_close()

    def setUp(self):
        if not self.server_available:
            self.skipTest("Local loopback networking is restricted in current sandbox environment")

    def test_get_dashboard_html(self):
        url = f"http://127.0.0.1:{self.port}/ui/dashboard.html"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            self.assertEqual(resp.status, 200)
            content_type = resp.headers.get("Content-Type", "")
            self.assertIn("text/html", content_type)
            body = resp.read().decode("utf-8")
            self.assertIn("Drosophila Connectome Swarm Mesh", body)
            self.assertIn("swarmCanvas", body)

    def test_get_index_html(self):
        url = f"http://127.0.0.1:{self.port}/ui/index.html"
        with urllib.request.urlopen(url, timeout=3.0) as resp:
            self.assertEqual(resp.status, 200)
            self.assertIn("text/html", resp.headers.get("Content-Type", ""))

    def test_get_static_plot_asset(self):
        url = f"http://127.0.0.1:{self.port}/plots/milestone6_fault_tolerance_and_jamming.png"
        with urllib.request.urlopen(url, timeout=3.0) as resp:
            self.assertEqual(resp.status, 200)
            self.assertEqual(resp.headers.get("Content-Type"), "image/png")
            data = resp.read()
            self.assertGreater(len(data), 1000)


def run_ui_tests() -> bool:
    """Executes all UI test cases and reports result."""
    print("\n" + "=" * 70)
    print("RUNNING FLYMESH UI & DASHBOARD AUTOMATED TESTS")
    print("=" * 70 + "\n")

    loader = unittest.defaultTestLoader
    suite = unittest.TestSuite()
    suite.addTest(loader.loadTestsFromTestCase(TestUIDashboardStatic))
    suite.addTest(loader.loadTestsFromTestCase(TestUIServerEndpoint))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_ui_tests()
    sys.exit(0 if success else 1)
