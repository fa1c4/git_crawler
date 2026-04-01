import argparse
import hashlib
import json
import logging
import os
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import requests
from pydantic import BaseModel, Field, ValidationError

try:
    from duckduckgo_search import DDGS
except ImportError:  # pragma: no cover - exercised through runtime validation
    DDGS = None


LOGGER = logging.getLogger(__name__)
SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[1]
DATA_DIR = REPO_ROOT / "data"
DEFAULT_INPUT_PATH = DATA_DIR / "c10k_github_repos_sorted.json"
DEFAULT_OUTPUT_PATH = DATA_DIR / "companies_repos.json"
DEFAULT_LEDGER_PATH = DATA_DIR / "companies_repos.checkpoint.jsonl"
DEFAULT_STATE_PATH = DATA_DIR / "companies_repos.state.json"
DEFAULT_MAX_SEARCH_RESULTS = 5
CHECKPOINT_FLUSH_INTERVAL = 25
MAX_EVIDENCE_RESULTS = 6
API_QUERY_PARAM = "key"

NON_COMPANY_KEYWORDS = {
    "foundation",
    "nonprofit",
    "non-profit",
    "university",
    "college",
    "institute",
    "academic",
    "laboratory",
    "laboratories",
    "research lab",
    "research center",
    "government",
    "gov",
    "ministry",
    "agency",
    "federal",
    "national laboratory",
    "association",
    "consortium",
    "council",
    "school",
}
NON_COMPANY_OWNERS = {
    "apache",
    "llvm",
    "kubernetes",
    "linuxfoundation",
    "freebsd",
    "gnome",
    "eclipse",
    "opencv",
}
IGNORE_COMPANY_CANDIDATES = {
    "github",
    "gitlab",
    "open source",
    "opensource",
    "repository",
    "repo",
    "project",
    "software",
    "community",
    "developers",
}
CORPORATE_SUFFIX_RE = re.compile(
    r"\b(inc|inc\.|llc|ltd|ltd\.|corp|corp\.|corporation|company|co\.|gmbh|ag|plc|limited)\b",
    re.IGNORECASE,
)


def _model_dump(model: BaseModel) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def _model_validate_json(model_cls: type[BaseModel], raw: str) -> BaseModel:
    if hasattr(model_cls, "model_validate_json"):
        return model_cls.model_validate_json(raw)
    return model_cls.parse_raw(raw)


class SearchHit(BaseModel):
    title: str = ""
    href: str = ""
    body: str = ""


class ClassificationResult(BaseModel):
    is_company_owned: bool
    company_name: str
    confidence: float = Field(ge=0.0, le=1.0)
    summary: str
    decision_source: str


class LedgerRecord(BaseModel):
    repo_full_name: str
    repo_url: str
    is_company_owned: bool
    company_name: str
    confidence: float = Field(ge=0.0, le=1.0)
    decision_source: str
    summary: str
    evidence: dict[str, Any]
    status: str
    error: Optional[str] = None
    updated_at: str
    input_fingerprint: str


class LLMClassificationPayload(BaseModel):
    is_company_owned: bool
    company_name: str
    confidence: float = Field(ge=0.0, le=1.0)
    summary: str


class APIConfig(BaseModel):
    base_url: str
    model: str
    api_key: str
    path_template: str = "/v1/chat/completions"
    auth: str = "bearer"
    timeout: float = 180.0

    @classmethod
    def from_env(cls) -> "APIConfig":
        missing = [
            name for name in ("API_BASE_URL", "API_MODEL", "API_KEY")
            if not os.getenv(name)
        ]
        if missing:
            raise ValueError(
                f"Missing required environment variables: {', '.join(missing)}"
            )
        return cls(
            base_url=os.environ["API_BASE_URL"].rstrip("/"),
            model=os.environ["API_MODEL"],
            api_key=os.environ["API_KEY"],
            path_template=os.getenv("API_PATH_TEMPLATE", "/v1/chat/completions"),
            auth=os.getenv("API_AUTH", "bearer").strip().lower(),
            timeout=float(os.getenv("API_TIMEOUT", "180")),
        )

    def endpoint(self) -> str:
        path = self.path_template.format(model=self.model)
        if not path.startswith("/"):
            path = f"/{path}"
        return f"{self.base_url}{path}"

    def headers_and_params(self) -> tuple[dict[str, str], dict[str, str]]:
        headers = {"Content-Type": "application/json"}
        params: dict[str, str] = {}
        if self.auth == "query":
            params[API_QUERY_PARAM] = self.api_key
        else:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers, params

    def public_state(self) -> dict[str, Any]:
        return {
            "base_url": self.base_url,
            "model": self.model,
            "path_template": self.path_template,
            "auth": self.auth,
            "timeout": self.timeout,
        }


class OpenAICompatibleClient:
    def __init__(self, config: APIConfig, session: Optional[requests.Session] = None):
        self.config = config
        self.session = session or requests.Session()

    def classify_repository(
        self,
        repo: dict[str, Any],
        evidence: dict[str, Any],
    ) -> ClassificationResult:
        system_prompt = (
            "You are a careful software industry analyst. "
            "Determine whether the GitHub repository is owned, created, or "
            "primarily maintained by a commercial company. "
            "Treat individuals, nonprofits, open-source foundations, academic "
            "institutions, and government organizations as not company-owned. "
            "Return only a JSON object with these keys: "
            "is_company_owned, company_name, confidence, summary. "
            "Use company_name='None' when not company-owned. "
            "Confidence must be between 0 and 1."
        )
        user_prompt = json.dumps(
            {
                "repo": repo,
                "evidence": evidence,
                "rules": {
                    "prefer_exact_repo_url_matches": True,
                    "exclude_nonprofit_foundation_academic_government": True,
                    "prefer_common_company_brand_name": True,
                },
            },
            ensure_ascii=False,
        )
        raw_text = self._request_chat_completion(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )
        try:
            return self._parse_classification(raw_text)
        except (ValidationError, ValueError):
            repair_prompt = (
                "Rewrite the following content into valid JSON only. "
                "Keep the same meaning and use exactly these keys: "
                "is_company_owned, company_name, confidence, summary.\n\n"
                f"{raw_text}"
            )
            repaired_text = self._request_chat_completion(
                [
                    {"role": "system", "content": "Return valid JSON only."},
                    {"role": "user", "content": repair_prompt},
                ]
            )
            return self._parse_classification(repaired_text)

    def _request_chat_completion(self, messages: list[dict[str, Any]]) -> str:
        headers, params = self.config.headers_and_params()
        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": 0,
        }
        response = self.session.post(
            self.config.endpoint(),
            headers=headers,
            params=params,
            json=payload,
            timeout=self.config.timeout,
        )
        response.raise_for_status()
        return self._extract_message_text(response.json())

    @staticmethod
    def _extract_message_text(response_json: dict[str, Any]) -> str:
        choices = response_json.get("choices", [])
        if not choices:
            raise ValueError("Chat completion response did not include choices.")
        message = choices[0].get("message", {})
        content = message.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(
                part.get("text", "")
                for part in content
                if isinstance(part, dict) and isinstance(part.get("text"), str)
            )
        raise ValueError("Unsupported chat completion message content.")

    def _parse_classification(self, raw_text: str) -> ClassificationResult:
        payload = _extract_json_object(raw_text)
        parsed = _model_validate_json(LLMClassificationPayload, payload)
        data = _model_dump(parsed)
        data["company_name"] = _normalize_company_name(data["company_name"])
        if not data["is_company_owned"]:
            data["company_name"] = "None"
        data["decision_source"] = "llm"
        return ClassificationResult(**data)


class DuckDuckGoSearcher:
    def __init__(self) -> None:
        if DDGS is None:
            raise ImportError(
                "duckduckgo-search is required to run company detection. "
                "Install dependencies from requirements.txt first."
            )

    def search(self, query: str, max_results: int) -> list[SearchHit]:
        ddgs = DDGS()
        try:
            raw_results = ddgs.text(query, max_results=max_results)
            results = list(raw_results or [])
        finally:
            close = getattr(ddgs, "close", None)
            if callable(close):
                close()
        hits = []
        for result in results:
            if not isinstance(result, dict):
                continue
            hits.append(
                SearchHit(
                    title=str(result.get("title", "") or ""),
                    href=str(result.get("href", "") or ""),
                    body=str(result.get("body", "") or ""),
                )
            )
        return hits


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Classify GitHub repositories by company ownership."
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT_PATH))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--ledger", default=str(DEFAULT_LEDGER_PATH))
    parser.add_argument("--state", default=str(DEFAULT_STATE_PATH))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume from checkpoint ledger if present.",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Discard prior checkpoint, state, and output files before starting.",
    )
    parser.add_argument(
        "--rerun-failure",
        action="store_true",
        help="When resuming, skip only status='ok' ledger entries and retry status='error' entries.",
    )
    parser.add_argument(
        "--max-search-results",
        type=int,
        default=DEFAULT_MAX_SEARCH_RESULTS,
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
    )


def resolve_default_paths(script_path: Optional[Path] = None) -> dict[str, Path]:
    root = (script_path or SCRIPT_PATH).resolve().parents[1]
    data_dir = root / "data"
    return {
        "repo_root": root,
        "input": data_dir / "c10k_github_repos_sorted.json",
        "output": data_dir / "companies_repos.json",
        "ledger": data_dir / "companies_repos.checkpoint.jsonl",
        "state": data_dir / "companies_repos.state.json",
    }


def fingerprint_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def retry_call(
    func: Callable[[], Any],
    attempts: int = 3,
    initial_delay: float = 1.0,
    backoff: float = 2.0,
    retryable: tuple[type[BaseException], ...] = (Exception,),
) -> Any:
    delay = initial_delay
    last_error: Optional[BaseException] = None
    for attempt in range(1, attempts + 1):
        try:
            return func()
        except retryable as exc:  # pragma: no branch - tiny helper
            last_error = exc
            if attempt == attempts:
                break
            LOGGER.warning("Retrying after error on attempt %s/%s: %s", attempt, attempts, exc)
            time.sleep(delay)
            delay *= backoff
    raise last_error or RuntimeError("retry_call failed without exception.")


def _safe_repo_full_name(repo: dict[str, Any]) -> str:
    owner = str(repo.get("owner", "") or "").strip()
    repo_name = str(repo.get("repo_name", "") or "").strip()
    return f"{owner}/{repo_name}".strip("/")


def _extract_json_object(raw_text: str) -> str:
    candidate = raw_text.strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*", "", candidate)
        candidate = re.sub(r"\s*```$", "", candidate)
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in model response.")
    return candidate[start:end + 1]


def _normalize_company_name(name: str) -> str:
    cleaned = (name or "").strip()
    if not cleaned or cleaned.lower() in {"none", "null", "n/a", "unknown"}:
        return "None"
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = CORPORATE_SUFFIX_RE.sub("", cleaned).strip(" ,.-")
    return cleaned or "None"


def build_search_queries(repo: dict[str, Any]) -> list[str]:
    full_name = _safe_repo_full_name(repo)
    url = str(repo.get("url", "") or "").strip()
    owner = str(repo.get("owner", "") or "").strip()
    queries = []
    if url:
        queries.append(f"\"{url}\" owner company")
    if full_name:
        queries.append(f"\"{full_name}\" github company owner")
    if owner:
        queries.append(f"site:github.com/{owner} {owner} company open source")
    return queries


def gather_evidence(
    repo: dict[str, Any],
    searcher: DuckDuckGoSearcher,
    max_search_results: int,
) -> dict[str, Any]:
    seen: set[tuple[str, str]] = set()
    evidence_results: list[dict[str, str]] = []
    queries_run: list[str] = []
    for query in build_search_queries(repo):
        queries_run.append(query)
        hits = retry_call(
            lambda current_query=query: searcher.search(current_query, max_search_results),
            attempts=3,
            initial_delay=1.0,
            backoff=2.0,
        )
        for hit in hits:
            key = (hit.href, hit.title)
            if key in seen:
                continue
            seen.add(key)
            evidence_results.append(_model_dump(hit))
            if len(evidence_results) >= MAX_EVIDENCE_RESULTS:
                break
        if len(evidence_results) >= MAX_EVIDENCE_RESULTS:
            break
    return {
        "queries": queries_run,
        "results": evidence_results,
    }


def _owner_is_non_company(repo: dict[str, Any]) -> bool:
    owner = str(repo.get("owner", "") or "").lower()
    return owner in NON_COMPANY_OWNERS or any(keyword in owner for keyword in NON_COMPANY_KEYWORDS)


def _non_company_signal_text(repo: dict[str, Any], evidence: dict[str, Any]) -> str:
    chunks = [
        str(repo.get("owner", "") or ""),
        str(repo.get("repo_name", "") or ""),
        str(repo.get("url", "") or ""),
    ]
    for result in evidence.get("results", []):
        chunks.append(str(result.get("title", "") or ""))
        chunks.append(str(result.get("body", "") or ""))
        chunks.append(str(result.get("href", "") or ""))
    return " ".join(chunks).lower()


def _extract_company_candidates(evidence: dict[str, Any]) -> Counter[str]:
    candidates: Counter[str] = Counter()
    patterns = [
        r"official (?:github )?repository (?:of|for) ([A-Z][A-Za-z0-9&.\- ]{1,60}?)(?= for\b| on\b|,|\.|$)",
        r"open[- ]source (?:project|repository) (?:by|from) ([A-Z][A-Za-z0-9&.\- ]{1,60}?)(?= for\b| on\b|,|\.|$)",
        r"maintained by ([A-Z][A-Za-z0-9&.\- ]{1,60}?)(?= for\b| on\b|,|\.|$)",
        r"from ([A-Z][A-Za-z0-9&.\- ]{1,60}?)(?= for\b| on\b|,|\.|$)",
    ]
    for result in evidence.get("results", []):
        combined = " ".join(
            [
                str(result.get("title", "") or ""),
                str(result.get("body", "") or ""),
            ]
        )
        for pattern in patterns:
            for match in re.findall(pattern, combined):
                candidate = _normalize_company_name(match)
                if candidate == "None":
                    continue
                if candidate.lower() in IGNORE_COMPANY_CANDIDATES:
                    continue
                if any(keyword in candidate.lower() for keyword in NON_COMPANY_KEYWORDS):
                    continue
                candidates[candidate] += 1
    return candidates


def classify_with_rules(repo: dict[str, Any], evidence: dict[str, Any]) -> Optional[ClassificationResult]:
    signals_text = _non_company_signal_text(repo, evidence)
    company_candidates = _extract_company_candidates(evidence)
    if _owner_is_non_company(repo) and not company_candidates:
        return ClassificationResult(
            is_company_owned=False,
            company_name="None",
            confidence=0.97,
            summary="Owner name strongly matches a foundation, academic, or other non-commercial organization.",
            decision_source="rule:owner_non_company",
        )
    if any(keyword in signals_text for keyword in NON_COMPANY_KEYWORDS) and not company_candidates:
        return ClassificationResult(
            is_company_owned=False,
            company_name="None",
            confidence=0.88,
            summary="Search evidence points to a foundation, academic, government, or other non-commercial organization.",
            decision_source="rule:evidence_non_company",
        )
    if company_candidates:
        top_company, count = company_candidates.most_common(1)[0]
        strong_official_signal = any(
            f"official repository of {top_company.lower()}" in signals_text
            or f"official github repository of {top_company.lower()}" in signals_text
            for _ in [0]
        )
        if count >= 2 or strong_official_signal:
            return ClassificationResult(
                is_company_owned=True,
                company_name=top_company,
                confidence=0.9,
                summary=f"Multiple search results consistently associate the repository with {top_company}.",
                decision_source="rule:company_evidence",
            )
    return None


def classify_repository(
    repo: dict[str, Any],
    searcher: DuckDuckGoSearcher,
    api_client: OpenAICompatibleClient,
    max_search_results: int,
) -> tuple[ClassificationResult, dict[str, Any]]:
    evidence = gather_evidence(repo, searcher, max_search_results)
    heuristic_result = classify_with_rules(repo, evidence)
    if heuristic_result is not None:
        return heuristic_result, evidence
    llm_result = retry_call(
        lambda: api_client.classify_repository(repo, evidence),
        attempts=3,
        initial_delay=2.0,
        backoff=2.0,
        retryable=(requests.RequestException, ValidationError, ValueError),
    )
    return llm_result, evidence


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def write_json(path: Path, payload: Any) -> None:
    ensure_parent_dir(path)
    with path.open("w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, indent=2, ensure_ascii=False)
        file_obj.write("\n")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    ensure_parent_dir(path)
    with path.open("a", encoding="utf-8") as file_obj:
        file_obj.write(json.dumps(payload, ensure_ascii=False) + "\n")


def read_ledger(path: Path) -> list[LedgerRecord]:
    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8") as file_obj:
        for line_number, line in enumerate(file_obj, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                records.append(LedgerRecord(**json.loads(stripped)))
            except (json.JSONDecodeError, ValidationError) as exc:
                raise ValueError(
                    f"Invalid checkpoint ledger entry at line {line_number}: {exc}"
                ) from exc
    return records


def build_grouped_output(records: Iterable[LedgerRecord]) -> dict[str, list[str]]:
    grouped: defaultdict[str, list[str]] = defaultdict(list)
    for record in records:
        if record.status != "ok":
            continue
        if not record.is_company_owned:
            continue
        company_name = _normalize_company_name(record.company_name)
        if company_name == "None":
            continue
        grouped[company_name].append(record.repo_full_name)
    output: dict[str, list[str]] = {}
    for company_name in sorted(grouped):
        output[company_name] = sorted(set(grouped[company_name]))
    return output


def write_state(
    path: Path,
    *,
    input_path: Path,
    input_fingerprint: str,
    total_repositories: int,
    completed_repositories: int,
    api_config: APIConfig,
    output_path: Path,
    ledger_path: Path,
) -> None:
    payload = {
        "input_path": str(input_path),
        "input_fingerprint": input_fingerprint,
        "total_repositories": total_repositories,
        "completed_repositories": completed_repositories,
        "last_updated_at": utc_now_iso(),
        "api_config": api_config.public_state(),
        "output_path": str(output_path),
        "ledger_path": str(ledger_path),
    }
    write_json(path, payload)


def prepare_run_files(
    *,
    fresh: bool,
    resume: bool,
    output_path: Path,
    ledger_path: Path,
    state_path: Path,
) -> None:
    if not fresh and resume:
        return
    for path in (output_path, ledger_path, state_path):
        if path.exists():
            path.unlink()


def validate_resume_state(
    *,
    resume: bool,
    fresh: bool,
    current_fingerprint: str,
    ledger_records: list[LedgerRecord],
    state_path: Path,
) -> None:
    if fresh or not resume:
        return
    seen_fingerprints = {record.input_fingerprint for record in ledger_records}
    if len(seen_fingerprints) > 1:
        raise RuntimeError("Checkpoint ledger mixes multiple input fingerprints; use --fresh.")
    if seen_fingerprints and current_fingerprint not in seen_fingerprints:
        raise RuntimeError("Checkpoint ledger does not match the current input file; use --fresh.")
    if state_path.exists():
        state_payload = load_json(state_path)
        previous_fingerprint = state_payload.get("input_fingerprint")
        if previous_fingerprint and previous_fingerprint != current_fingerprint:
            raise RuntimeError("State file does not match the current input file; use --fresh.")


def build_processed_names(
    ledger_records: Iterable[LedgerRecord],
    *,
    rerun_failure: bool,
) -> set[str]:
    if not rerun_failure:
        return {record.repo_full_name for record in ledger_records}
    return {
        record.repo_full_name
        for record in ledger_records
        if record.status == "ok"
    }


def process_repositories(
    *,
    input_path: Path,
    output_path: Path,
    ledger_path: Path,
    state_path: Path,
    api_config: APIConfig,
    max_search_results: int,
    limit: Optional[int] = None,
    resume: bool = True,
    fresh: bool = False,
    rerun_failure: bool = False,
    searcher: Optional[DuckDuckGoSearcher] = None,
    api_client: Optional[OpenAICompatibleClient] = None,
) -> dict[str, Any]:
    prepare_run_files(
        fresh=fresh,
        resume=resume,
        output_path=output_path,
        ledger_path=ledger_path,
        state_path=state_path,
    )
    input_data = load_json(input_path)
    repositories = input_data.get("repositories", [])
    if not isinstance(repositories, list):
        raise ValueError("Input JSON must contain a list under the 'repositories' key.")
    if limit is not None:
        repositories = repositories[:limit]
    input_fingerprint = fingerprint_file(input_path)
    ledger_records = read_ledger(ledger_path) if resume and not fresh else []
    validate_resume_state(
        resume=resume,
        fresh=fresh,
        current_fingerprint=input_fingerprint,
        ledger_records=ledger_records,
        state_path=state_path,
    )
    processed_names = build_processed_names(
        ledger_records,
        rerun_failure=rerun_failure,
    )
    searcher = searcher or DuckDuckGoSearcher()
    api_client = api_client or OpenAICompatibleClient(api_config)
    total = len(repositories)
    completed_records = list(ledger_records)
    newly_processed = 0

    LOGGER.info("Loaded %s repositories from %s", total, input_path)
    for index, repo in enumerate(repositories, start=1):
        repo_full_name = _safe_repo_full_name(repo)
        repo_url = str(repo.get("url", "") or "")
        if not repo_full_name:
            LOGGER.warning("Skipping repository at index %s because owner/repo_name is missing.", index)
            continue
        if repo_full_name in processed_names:
            LOGGER.info("[%s/%s] Skipping %s from checkpoint.", index, total, repo_full_name)
            continue

        LOGGER.info("[%s/%s] Processing %s", index, total, repo_full_name)
        try:
            classification, evidence = classify_repository(
                repo,
                searcher=searcher,
                api_client=api_client,
                max_search_results=max_search_results,
            )
            record = LedgerRecord(
                repo_full_name=repo_full_name,
                repo_url=repo_url,
                status="ok",
                error=None,
                updated_at=utc_now_iso(),
                input_fingerprint=input_fingerprint,
                evidence=evidence,
                **_model_dump(classification),
            )
        except Exception as exc:  # pragma: no cover - integration-oriented branch
            LOGGER.exception("Failed to process %s", repo_full_name)
            record = LedgerRecord(
                repo_full_name=repo_full_name,
                repo_url=repo_url,
                is_company_owned=False,
                company_name="None",
                confidence=0.0,
                decision_source="error",
                summary="Processing failed before a classification could be produced.",
                evidence={"queries": build_search_queries(repo), "results": []},
                status="error",
                error=str(exc),
                updated_at=utc_now_iso(),
                input_fingerprint=input_fingerprint,
            )

        append_jsonl(ledger_path, _model_dump(record))
        completed_records.append(record)
        processed_names.add(repo_full_name)
        newly_processed += 1

        if newly_processed % CHECKPOINT_FLUSH_INTERVAL == 0:
            grouped_output = build_grouped_output(completed_records)
            write_json(output_path, grouped_output)
            write_state(
                state_path,
                input_path=input_path,
                input_fingerprint=input_fingerprint,
                total_repositories=total,
                completed_repositories=len(completed_records),
                api_config=api_config,
                output_path=output_path,
                ledger_path=ledger_path,
            )

    grouped_output = build_grouped_output(completed_records)
    write_json(output_path, grouped_output)
    write_state(
        state_path,
        input_path=input_path,
        input_fingerprint=input_fingerprint,
        total_repositories=total,
        completed_repositories=len(completed_records),
        api_config=api_config,
        output_path=output_path,
        ledger_path=ledger_path,
    )
    LOGGER.info("Saved %s companies to %s", len(grouped_output), output_path)
    return {
        "total_repositories": total,
        "completed_repositories": len(completed_records),
        "newly_processed": newly_processed,
        "companies_found": len(grouped_output),
        "output_path": str(output_path),
        "ledger_path": str(ledger_path),
        "state_path": str(state_path),
    }


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    configure_logging(args.log_level)
    default_paths = resolve_default_paths()
    input_path = Path(args.input or default_paths["input"])
    output_path = Path(args.output or default_paths["output"])
    ledger_path = Path(args.ledger or default_paths["ledger"])
    state_path = Path(args.state or default_paths["state"])
    api_config = APIConfig.from_env()

    summary = process_repositories(
        input_path=input_path,
        output_path=output_path,
        ledger_path=ledger_path,
        state_path=state_path,
        api_config=api_config,
        max_search_results=args.max_search_results,
        limit=args.limit,
        resume=args.resume,
        fresh=args.fresh,
        rerun_failure=args.rerun_failure,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
