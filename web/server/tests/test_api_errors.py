"""Errors under /api/ come back as JSON, whatever raised them."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cistem_server  # noqa: E402


def _boom():
    raise RuntimeError("the database is on fire")


def _inf():
    return cistem_server.jsonify({"beam_tilt_x_mrad": float("inf")})


cistem_server.app.add_url_rule("/api/_test_boom", "test_boom", _boom)
cistem_server.app.add_url_rule("/api/_test_inf", "test_inf", _inf)
cistem_server.app.add_url_rule("/_test_boom_page", "test_boom_page", _boom)


class ApiErrorTests(unittest.TestCase):
    def setUp(self):
        self.client = cistem_server.app.test_client()

    def test_an_unhandled_exception_is_json_with_its_message(self):
        r = self.client.get("/api/_test_boom")
        self.assertEqual(r.status_code, 500)
        self.assertEqual(r.mimetype, "application/json")
        self.assertIn("RuntimeError: the database is on fire", r.get_json()["error"])

    def test_a_non_finite_float_is_a_500_not_a_bare_token(self):
        # json.dumps writes inf as the token Infinity, which no browser parses; the page then sat on
        # "Loading..." with the cause nowhere (a beam tilt divided by a Cs of 0, 2026-11-16).
        r = self.client.get("/api/_test_inf")
        self.assertEqual(r.status_code, 500)
        self.assertEqual(r.mimetype, "application/json")
        self.assertIn("not JSON compliant", r.get_json()["error"])

    def test_a_missing_api_route_is_json(self):
        r = self.client.get("/api/no/such/route")
        self.assertEqual(r.status_code, 404)
        self.assertEqual(r.mimetype, "application/json")
        self.assertTrue(r.get_json()["error"])

    def test_pages_keep_html_errors(self):
        r = self.client.get("/_test_boom_page")
        self.assertEqual(r.status_code, 500)
        self.assertEqual(r.mimetype, "text/html")


if __name__ == "__main__":
    unittest.main()
