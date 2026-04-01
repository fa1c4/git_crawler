import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock


MODULE_PATH = Path("/data/zym/git_crawler/company_agent/finding_company_agent.py")
SPEC = importlib.util.spec_from_file_location("finding_company_agent", MODULE_PATH)
finding_company_agent = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(finding_company_agent)


class FakeSearchClient:
    def __init__(self, mapping):
        self.mapping = mapping
        self.queries = []

    def search(self, query, max_results):
        self.queries.append((query, max_results))
        return [
            finding_company_agent.SearchHit(**item)
            for item in self.mapping.get(query, [])
        ]


class FakeApiClient:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def classify_repository(self, repo, evidence):
        self.calls.append((repo, evidence))
        return self.result


class FindingCompanyAgentTests(unittest.TestCase):
    def test_api_config_from_env_and_endpoint(self):
        with mock.patch.dict(
            "os.environ",
            {
                "API_BASE_URL": "https://example.com/",
                "API_MODEL": "gpt-5.2",
                "API_KEY": "test-key",
                "API_PATH_TEMPLATE": "/v1/chat/completions",
                "API_AUTH": "bearer",
            },
            clear=True,
        ):
            config = finding_company_agent.APIConfig.from_env()

        self.assertEqual("https://example.com/v1/chat/completions", config.endpoint())
        headers, params = config.headers_and_params()
        self.assertEqual("Bearer test-key", headers["Authorization"])
        self.assertEqual({}, params)

    def test_query_auth_moves_api_key_to_params(self):
        config = finding_company_agent.APIConfig(
            base_url="https://example.com",
            model="gpt-5.2",
            api_key="test-key",
            auth="query",
        )

        headers, params = config.headers_and_params()

        self.assertNotIn("Authorization", headers)
        self.assertEqual({"key": "test-key"}, params)

    def test_resolve_default_paths_uses_git_crawler_root(self):
        fake_script = Path("/tmp/somewhere/git_crawler/company_agent/finding_company_agent.py")

        paths = finding_company_agent.resolve_default_paths(fake_script)

        self.assertEqual(Path("/tmp/somewhere/git_crawler"), paths["repo_root"])
        self.assertEqual(
            Path("/tmp/somewhere/git_crawler/data/companies_repos.json"),
            paths["output"],
        )

    def test_rule_classifier_marks_foundation_as_non_company(self):
        repo = {"owner": "apache", "repo_name": "kafka", "url": "https://github.com/apache/kafka"}
        evidence = {
            "queries": [],
            "results": [
                {
                    "title": "Apache Kafka - Apache Software Foundation",
                    "href": "https://kafka.apache.org",
                    "body": "Apache Kafka is an open-source project of the Apache Software Foundation.",
                }
            ],
        }

        result = finding_company_agent.classify_with_rules(repo, evidence)

        self.assertIsNotNone(result)
        self.assertFalse(result.is_company_owned)
        self.assertEqual("None", result.company_name)

    def test_rule_classifier_marks_repeated_company_signals_as_company_owned(self):
        repo = {"owner": "vercel", "repo_name": "next.js", "url": "https://github.com/vercel/next.js"}
        evidence = {
            "queries": [],
            "results": [
                {
                    "title": "Official repository of Vercel",
                    "href": "https://vercel.com/open-source",
                    "body": "Official GitHub repository of Vercel for Next.js.",
                },
                {
                    "title": "Open-source project by Vercel",
                    "href": "https://nextjs.org",
                    "body": "Next.js is an open-source project by Vercel.",
                },
            ],
        }

        result = finding_company_agent.classify_with_rules(repo, evidence)

        self.assertIsNotNone(result)
        self.assertTrue(result.is_company_owned)
        self.assertEqual("Vercel", result.company_name)

    def test_ambiguous_repo_falls_back_to_llm_client(self):
        repo = {"owner": "someone", "repo_name": "tool", "url": "https://github.com/someone/tool"}
        searcher = FakeSearchClient(
            {
                "\"https://github.com/someone/tool\" owner company": [
                    {
                        "title": "GitHub - someone/tool",
                        "href": "https://github.com/someone/tool",
                        "body": "A useful tool.",
                    }
                ],
                "\"someone/tool\" github company owner": [],
                "site:github.com/someone someone company open source": [],
            }
        )
        api_client = FakeApiClient(
            finding_company_agent.ClassificationResult(
                is_company_owned=True,
                company_name="Acme",
                confidence=0.74,
                summary="Search evidence links the project to Acme.",
                decision_source="llm",
            )
        )

        result, evidence = finding_company_agent.classify_repository(
            repo,
            searcher=searcher,
            api_client=api_client,
            max_search_results=3,
        )

        self.assertEqual("Acme", result.company_name)
        self.assertEqual(1, len(api_client.calls))
        self.assertTrue(evidence["results"])

    def test_resume_skips_completed_repos_and_rebuilds_output(self):
        config = finding_company_agent.APIConfig(
            base_url="https://example.com",
            model="gpt-5.2",
            api_key="test-key",
        )
        api_client = FakeApiClient(
            finding_company_agent.ClassificationResult(
                is_company_owned=True,
                company_name="Acme",
                confidence=0.8,
                summary="Owned by Acme.",
                decision_source="llm",
            )
        )
        searcher = FakeSearchClient(
            {
                "\"https://github.com/acme/tool\" owner company": [],
                "\"acme/tool\" github company owner": [],
                "site:github.com/acme acme company open source": [],
            }
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            input_path = temp_root / "input.json"
            output_path = temp_root / "companies.json"
            ledger_path = temp_root / "companies.jsonl"
            state_path = temp_root / "companies.state.json"
            payload = {
                "repositories": [
                    {
                        "owner": "vercel",
                        "repo_name": "next.js",
                        "url": "https://github.com/vercel/next.js",
                    },
                    {
                        "owner": "acme",
                        "repo_name": "tool",
                        "url": "https://github.com/acme/tool",
                    },
                ]
            }
            input_path.write_text(json.dumps(payload), encoding="utf-8")
            fingerprint = finding_company_agent.fingerprint_file(input_path)
            existing_record = finding_company_agent.LedgerRecord(
                repo_full_name="vercel/next.js",
                repo_url="https://github.com/vercel/next.js",
                is_company_owned=True,
                company_name="Vercel",
                confidence=0.91,
                decision_source="rule:company_evidence",
                summary="Multiple search results consistently associate the repository with Vercel.",
                evidence={"queries": [], "results": []},
                status="ok",
                error=None,
                updated_at=finding_company_agent.utc_now_iso(),
                input_fingerprint=fingerprint,
            )
            ledger_path.write_text(
                json.dumps(finding_company_agent._model_dump(existing_record)) + "\n",
                encoding="utf-8",
            )

            summary = finding_company_agent.process_repositories(
                input_path=input_path,
                output_path=output_path,
                ledger_path=ledger_path,
                state_path=state_path,
                api_config=config,
                max_search_results=3,
                resume=True,
                searcher=searcher,
                api_client=api_client,
            )

            output_payload = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(1, summary["newly_processed"])
        self.assertEqual(["acme/tool"], output_payload["Acme"])
        self.assertEqual(["vercel/next.js"], output_payload["Vercel"])

    def test_resume_detects_input_fingerprint_mismatch(self):
        config = finding_company_agent.APIConfig(
            base_url="https://example.com",
            model="gpt-5.2",
            api_key="test-key",
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            input_path = temp_root / "input.json"
            output_path = temp_root / "companies.json"
            ledger_path = temp_root / "companies.jsonl"
            state_path = temp_root / "companies.state.json"
            input_path.write_text(
                json.dumps({"repositories": [{"owner": "acme", "repo_name": "tool", "url": "https://github.com/acme/tool"}]}),
                encoding="utf-8",
            )
            wrong_record = finding_company_agent.LedgerRecord(
                repo_full_name="acme/tool",
                repo_url="https://github.com/acme/tool",
                is_company_owned=True,
                company_name="Acme",
                confidence=0.7,
                decision_source="llm",
                summary="Owned by Acme.",
                evidence={"queries": [], "results": []},
                status="ok",
                error=None,
                updated_at=finding_company_agent.utc_now_iso(),
                input_fingerprint="not-the-current-fingerprint",
            )
            ledger_path.write_text(
                json.dumps(finding_company_agent._model_dump(wrong_record)) + "\n",
                encoding="utf-8",
            )

            with self.assertRaises(RuntimeError):
                finding_company_agent.process_repositories(
                    input_path=input_path,
                    output_path=output_path,
                    ledger_path=ledger_path,
                    state_path=state_path,
                    api_config=config,
                    max_search_results=3,
                    resume=True,
                    searcher=FakeSearchClient({}),
                    api_client=FakeApiClient(
                        finding_company_agent.ClassificationResult(
                            is_company_owned=False,
                            company_name="None",
                            confidence=0.5,
                            summary="No company.",
                            decision_source="llm",
                        )
                    ),
                )

    def test_rerun_failure_retries_error_entries_but_skips_ok_entries(self):
        config = finding_company_agent.APIConfig(
            base_url="https://example.com",
            model="gpt-5.2",
            api_key="test-key",
        )
        api_client = FakeApiClient(
            finding_company_agent.ClassificationResult(
                is_company_owned=True,
                company_name="Acme",
                confidence=0.83,
                summary="Recovered successfully on retry.",
                decision_source="llm",
            )
        )
        searcher = FakeSearchClient(
            {
                "\"https://github.com/acme/tool\" owner company": [],
                "\"acme/tool\" github company owner": [],
                "site:github.com/acme acme company open source": [],
            }
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            input_path = temp_root / "input.json"
            output_path = temp_root / "companies.json"
            ledger_path = temp_root / "companies.jsonl"
            state_path = temp_root / "companies.state.json"
            payload = {
                "repositories": [
                    {
                        "owner": "vercel",
                        "repo_name": "next.js",
                        "url": "https://github.com/vercel/next.js",
                    },
                    {
                        "owner": "acme",
                        "repo_name": "tool",
                        "url": "https://github.com/acme/tool",
                    },
                ]
            }
            input_path.write_text(json.dumps(payload), encoding="utf-8")
            fingerprint = finding_company_agent.fingerprint_file(input_path)
            ok_record = finding_company_agent.LedgerRecord(
                repo_full_name="vercel/next.js",
                repo_url="https://github.com/vercel/next.js",
                is_company_owned=True,
                company_name="Vercel",
                confidence=0.91,
                decision_source="rule:company_evidence",
                summary="Multiple search results consistently associate the repository with Vercel.",
                evidence={"queries": [], "results": []},
                status="ok",
                error=None,
                updated_at=finding_company_agent.utc_now_iso(),
                input_fingerprint=fingerprint,
            )
            error_record = finding_company_agent.LedgerRecord(
                repo_full_name="acme/tool",
                repo_url="https://github.com/acme/tool",
                is_company_owned=False,
                company_name="None",
                confidence=0.0,
                decision_source="error",
                summary="Processing failed before a classification could be produced.",
                evidence={"queries": [], "results": []},
                status="error",
                error="temporary failure",
                updated_at=finding_company_agent.utc_now_iso(),
                input_fingerprint=fingerprint,
            )
            ledger_path.write_text(
                "\n".join(
                    [
                        json.dumps(finding_company_agent._model_dump(ok_record)),
                        json.dumps(finding_company_agent._model_dump(error_record)),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            summary = finding_company_agent.process_repositories(
                input_path=input_path,
                output_path=output_path,
                ledger_path=ledger_path,
                state_path=state_path,
                api_config=config,
                max_search_results=3,
                resume=True,
                rerun_failure=True,
                searcher=searcher,
                api_client=api_client,
            )
            output_payload = json.loads(output_path.read_text(encoding="utf-8"))
            ledger_lines = [
                json.loads(line)
                for line in ledger_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

        self.assertEqual(1, summary["newly_processed"])
        self.assertEqual(1, len(api_client.calls))
        self.assertEqual(["acme/tool"], output_payload["Acme"])
        self.assertEqual(["vercel/next.js"], output_payload["Vercel"])
        self.assertEqual(3, len(ledger_lines))
        self.assertEqual("error", ledger_lines[1]["status"])
        self.assertEqual("ok", ledger_lines[2]["status"])


if __name__ == "__main__":
    unittest.main()
