import unittest

import sell_stress as ss


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class FakeHTMLScraper:
    def __init__(self, values):
        self.values = values
        self.calls = 0

    def fetch_index_page(self):
        self.calls += 1
        return self.values


class FreeFloatResolverTests(unittest.TestCase):
    def test_uses_description_freefloat_alias_and_decimal_comma(self):
        def request_get(url, timeout=10):
            if "/description" in url:
                return FakeResponse({
                    "description": {
                        "columns": ["name", "title", "value"],
                        "data": [["FREEFLOAT", "Free float", "32,5"], ["ISSUESIZE", "Issue size", 1000]],
                    }
                })
            return FakeResponse({})

        client = ss.MoexISSClient(request_get=request_get, retries=1, pause=0)
        resolver = ss.FreeFloatResolverV2(client, FakeHTMLScraper({}))
        payload = resolver.get("TEST")

        self.assertAlmostEqual(payload["free_float"], 0.325)
        self.assertEqual(payload["source"], "ISS_DESCRIPTION")

    def test_html_fallback_is_downloaded_once_for_batch(self):
        def request_get(url, timeout=10):
            return FakeResponse({})

        scraper = FakeHTMLScraper({"AAA": 0.25, "BBB": 0.4})
        client = ss.MoexISSClient(request_get=request_get, retries=1, pause=0)
        resolver = ss.FreeFloatResolverV2(client, scraper)

        result = resolver.get_many(["AAA", "BBB"])

        self.assertAlmostEqual(result["AAA"]["free_float"], 0.25)
        self.assertAlmostEqual(result["BBB"]["free_float"], 0.4)
        self.assertEqual(scraper.calls, 1)
        self.assertEqual(result["AAA"]["source"], "MOEX_LISTING")

    def test_official_listing_table_parses_percent_and_paginates(self):
        calls = []

        class HtmlResponse:
            def __init__(self, text):
                self.text = text

            def raise_for_status(self):
                return None

        def request_get(url, params=None, headers=None, timeout=15):
            page = params["pagenumber"]
            calls.append(page)
            if page == 0:
                html = """<table><tr><th>№</th><th>Код</th><th>Наименование</th>
                <th>Коэффициент free-float, %</th></tr>
                <tr><td>1</td><td>AAA</td><td>Issuer A</td><td>32,5</td></tr></table>"""
            else:
                html = "<html><body>no more rows</body></html>"
            return HtmlResponse(html)

        scraper = ss.MoexHTMLFreeFloatScraper(request_get=request_get)
        values = scraper.fetch_listing_freefloat()

        self.assertAlmostEqual(values["AAA"], 0.325)
        self.assertEqual(calls, [0, 1])

    def test_missing_value_contains_diagnostic(self):
        def request_get(url, timeout=10):
            return FakeResponse({})

        client = ss.MoexISSClient(request_get=request_get, retries=1, pause=0)
        resolver = ss.FreeFloatResolverV2(client, FakeHTMLScraper({}))
        payload = resolver.get("UNKNOWN")

        self.assertIsNone(payload["free_float"])
        self.assertEqual(payload["source"], "NOT_FOUND")
        self.assertTrue(payload["error"])


if __name__ == "__main__":
    unittest.main()
