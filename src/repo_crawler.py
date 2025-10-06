'''
Github Crawler for quality repositories collection (bucketed to bypass 1000-result Search API cap)
'''
import requests
import time
import json
import random
from datetime import datetime, timedelta
import os
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass, asdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('../logs/github_crawler.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class RepoData:
    """Data structure to hold repository information"""
    # Meta data
    repo_name: str
    url: str
    owner: str
    language: str
    license: str
    last_commit_date: str
    
    # Active data
    stars: int
    forks: int
    watchers: int
    issues_open: int
    issues_closed: int
    pr_open: int
    pr_closed: int
    contributors: int
    commits: int
    releases: int
    
    # Quality data
    readme_lines: int
    has_wiki: bool
    has_pages: bool
    ci_cd: bool

class GitHubCrawler:
    def __init__(self, tokens: List[str], min_stars: int = 100):
        """
        Initialize GitHub crawler with multiple tokens for rate limit management
        
        Args:
            tokens: List of GitHub personal access tokens
            min_stars: Minimum number of stars for repositories to crawl
        """
        self.tokens = tokens
        self.current_token_index = 0
        self.min_stars = min_stars
        self.base_url = "https://api.github.com"
        self.session = requests.Session()
        
        # Anti-ban settings
        self.min_delay = 0.8  # Minimum delay between requests (seconds)
        self.max_delay = 2.2  # Maximum delay between requests (seconds)
        self.rate_limit_buffer = 50  # Keep this many requests as buffer
        
        # Data storage
        self.repos_data: List[RepoData] = []
        self._last_rate_remaining: Optional[int] = None
        self._last_rate_reset_at: Optional[int] = None
        
    def get_headers(self) -> Dict[str, str]:
        """Get headers with current token"""
        return {
            'Authorization': f'token {self.tokens[self.current_token_index]}',
            'Accept': 'application/vnd.github.v3+json',
            'User-Agent': 'GitHub-Repo-Analyzer/1.0'
        }
    
    def rotate_token(self):
        """Rotate to next available token"""
        self.current_token_index = (self.current_token_index + 1) % len(self.tokens)
        logger.info(f"Rotated to token {self.current_token_index + 1}/{len(self.tokens)}")
    
    def _update_rate_from_headers(self, response: requests.Response):
        try:
            rem = response.headers.get('X-RateLimit-Remaining')
            reset = response.headers.get('X-RateLimit-Reset')
            if rem is not None:
                self._last_rate_remaining = int(rem)
            if reset is not None:
                self._last_rate_reset_at = int(reset)
            if self._last_rate_remaining is not None and self._last_rate_reset_at is not None:
                reset_time = datetime.fromtimestamp(self._last_rate_reset_at)
                logger.info(
                    f"Rate limit: {self._last_rate_remaining} requests remaining, resets at {reset_time}"
                )
        except Exception:
            pass

    def _should_pause_for_rate(self) -> bool:
        if self._last_rate_remaining is None:
            return False
        return self._last_rate_remaining <= self.rate_limit_buffer

    def check_rate_limit(self) -> bool:
        """(Kept) Explicitly check rate limit when necessary"""
        try:
            response = self.session.get(
                f"{self.base_url}/rate_limit",
                headers=self.get_headers()
            )
            if response.status_code == 200:
                data = response.json()
                remaining = data['resources']['core']['remaining']
                reset_time = datetime.fromtimestamp(data['resources']['core']['reset'])
                self._last_rate_remaining = remaining
                self._last_rate_reset_at = int(data['resources']['core']['reset'])
                logger.info(f"Rate limit: {remaining} requests remaining, resets at {reset_time}")
                if remaining <= self.rate_limit_buffer:
                    logger.warning(f"Approaching rate limit. {remaining} requests remaining.")
                    return False
                return True
            else:
                logger.error(f"Failed to check rate limit: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"Error checking rate limit: {e}")
            return False
    
    def wait_for_rate_limit_reset(self):
        """Wait until rate limit resets"""
        try:
            reset_at = self._last_rate_reset_at
            if reset_at is None:
                # Fallback to explicit fetch
                response = self.session.get(
                    f"{self.base_url}/rate_limit",
                    headers=self.get_headers()
                )
                if response.status_code == 200:
                    data = response.json()
                    reset_at = int(data['resources']['core']['reset'])
            if reset_at:
                wait_time = max(0.0, datetime.fromtimestamp(reset_at) - datetime.now()).total_seconds() + 60
                if wait_time > 0:
                    logger.info(f"Waiting {wait_time:.1f} seconds for rate limit reset...")
                    time.sleep(wait_time)
            else:
                time.sleep(600)
        except Exception as e:
            logger.error(f"Error waiting for rate limit reset: {e}")
            time.sleep(600)
    
    def make_request(self, url: str, params: Dict = None) -> Optional[requests.Response]:
        """Make API request with anti-ban mechanisms"""
        max_retries = len(self.tokens) * 2
        
        for attempt in range(max_retries):
            try:
                if self._should_pause_for_rate():
                    if len(self.tokens) > 1:
                        self.rotate_token()
                    else:
                        self.wait_for_rate_limit_reset()

                # Random delay to avoid being flagged as bot
                delay = random.uniform(self.min_delay, self.max_delay)
                time.sleep(delay)
                
                response = self.session.get(url, headers=self.get_headers(), params=params)
                self._update_rate_from_headers(response)
                
                if response.status_code == 200:
                    return response
                elif response.status_code == 403:
                    logger.warning("Rate limit exceeded or forbidden, rotating token or waiting...")
                    if len(self.tokens) > 1:
                        self.rotate_token()
                    else:
                        self.wait_for_rate_limit_reset()
                elif response.status_code == 404:
                    logger.warning(f"Resource not found: {url}")
                    return None
                else:
                    logger.error(f"API error {response.status_code}: {response.text}")
            except requests.exceptions.RequestException as e:
                logger.error(f"Request error (attempt {attempt + 1}): {e}")
                time.sleep(min(300, 30 * (attempt + 1)))
        
        logger.error(f"Failed to make request after {max_retries} attempts")
        return None

    # -------------------- Search helpers (NEW) --------------------
    @staticmethod
    def _parse_last_page_from_link(link_header: str) -> Optional[int]:
        """Robustly parse the last page number from GitHub Link header."""
        try:
            # Example: <...&page=2>; rel="next", <...&page=34>; rel="last"
            parts = link_header.split(',')
            for p in parts:
                if 'rel="last"' in p:
                    # extract page=number between ...page=NNN>
                    start = p.find('page=')
                    if start != -1:
                        start += len('page=')
                        end = p.find('>', start)
                        return int(p[start:end])
        except Exception:
            pass
        return None

    def _search_page(self, query: str, sort: str, page: int, per_page: int = 100) -> List[Dict]:
        params = {
            'q': query,
            'sort': sort,
            'order': 'desc',
            'per_page': per_page,
            'page': page
        }
        response = self.make_request(f"{self.base_url}/search/repositories", params)
        if not response:
            return []
        data = response.json()
        return data.get('items', [])

    def search_repositories(self, query: str, sort: str = "stars", per_page: int = 100) -> List[Dict]:
        """Single-query search with a practical hard cap (~1000)."""
        repos = []
        page = 1
        while True:
            items = self._search_page(query, sort, page, per_page)
            if not items:
                break
            repos.extend(items)
            logger.info(f"Fetched page {page}, total repos so far: {len(repos)} | query={query}")
            # natural API cap ~1000 (10 pages of 100) — stop to avoid wasted calls
            if len(items) < per_page or page >= 10:
                break
            page += 1
        return repos

    # NEW: bucketing by stars + (adaptive) date splitting
    @staticmethod
    def _star_buckets(min_star: int, max_star: int, step: int) -> List[Tuple[int, int]]:
        buckets: List[Tuple[int, int]] = []
        cur = min_star
        while cur <= max_star:
            buckets.append((cur, cur + step - 1))
            cur += step
        return buckets

    def _search_with_date_splitting(self, base_query: str, sort: str = 'stars', date_field: str = 'pushed', 
                                    start: str = '2014-01-01', end: str = None, step_days: int = 180,
                                    max_pages_threshold: int = 9) -> List[Dict]:
        """
        Split a query over date ranges to ensure each sub-query returns well under the 1000 cap.
        If a segment still appears too large (>= max_pages_threshold pages), recursively split further.
        """
        if end is None:
            end = datetime.utcnow().date().isoformat()
        s = datetime.fromisoformat(start)
        e = datetime.fromisoformat(end)
        results: List[Dict] = []
        seen_ids = set()
        
        def _daterange(sdt: datetime, edt: datetime, step: int):
            cur = sdt
            delta = timedelta(days=step)
            while cur < edt:
                nxt = min(edt, cur + delta)
                yield cur, nxt
                cur = nxt

        for ds, de in _daterange(s, e, step_days):
            q = f"{base_query} {date_field}:{ds.date()}..{de.date()}"
            # probe first to estimate size
            items = self.search_repositories(q, sort=sort, per_page=100)
            pages_est = (len(items) + 99) // 100
            if pages_est >= max_pages_threshold and (de - ds).days > 30:
                # too big — split further
                logger.info(f"Segment too large, splitting further: {q}")
                sub_items = self._search_with_date_splitting(
                    base_query, sort, date_field,
                    start=str(ds.date()), end=str(de.date()),
                    step_days=max(30, step_days // 2),
                    max_pages_threshold=max_pages_threshold
                )
                for it in sub_items:
                    if it['id'] not in seen_ids:
                        seen_ids.add(it['id'])
                        results.append(it)
            else:
                for it in items:
                    if it['id'] not in seen_ids:
                        seen_ids.add(it['id'])
                        results.append(it)
        return results

    def search_repositories_bucketing(self, language: str, min_stars: int) -> List[Dict]:
        """
        Bucketed search that bypasses the 1000-result cap by splitting on stars and (if needed) dates.
        """
        all_items: List[Dict] = []
        seen = set()
        # Heuristic star buckets: denser at low star ranges
        star_plan = [
            (min_stars, 1000, 100),
            (1001, 3000, 200),
            (3001, 10000, 500),
            (10001, 100000, 5000),
        ]
        for lo, hi, step in star_plan:
            lo = max(lo, min_stars)
            if lo > hi:
                continue
            for s_lo, s_hi in self._star_buckets(lo, hi, step):
                base_q = f"language:{language} stars:{s_lo}..{s_hi}"
                # First try direct query; if it looks large, fall back to date splitting
                items = self.search_repositories(base_q, sort='stars', per_page=100)
                if len(items) >= 900:  # heuristic: close to cap, split by date
                    items = self._search_with_date_splitting(
                        base_q, sort='stars', date_field='pushed', start='2014-01-01', step_days=180
                    )
                for it in items:
                    if it['id'] not in seen:
                        seen.add(it['id'])
                        all_items.append(it)
                logger.info(f"Bucket {language} stars:{s_lo}..{s_hi} -> cumulative {len(all_items)} repos")
        return all_items

    # -------------------- Original details collection --------------------
    def get_detailed_repo_info(self, repo_basic: Dict) -> Optional[RepoData]:
        """Get detailed information about a repository"""
        try:
            # Get repository details
            repo_response = self.make_request(f"{self.base_url}/repos/{repo_basic['full_name']}")
            if not repo_response:
                return None
                
            repo_data = repo_response.json()
            
            # Get issues count (open)
            issues_response = self.make_request(
                f"{self.base_url}/repos/{repo_basic['full_name']}/issues",
                {'state': 'open', 'per_page': 1}
            )
            issues_open = 0
            if issues_response and 'Link' in issues_response.headers:
                last_page = self._parse_last_page_from_link(issues_response.headers['Link'])
                if last_page:
                    issues_open = last_page
            
            # Get closed issues count
            issues_closed_response = self.make_request(
                f"{self.base_url}/repos/{repo_basic['full_name']}/issues",
                {'state': 'closed', 'per_page': 1}
            )
            issues_closed = 0
            if issues_closed_response and 'Link' in issues_closed_response.headers:
                last_page = self._parse_last_page_from_link(issues_closed_response.headers['Link'])
                if last_page:
                    issues_closed = last_page
            
            # Get pull requests count
            pr_response = self.make_request(
                f"{self.base_url}/repos/{repo_basic['full_name']}/pulls",
                {'state': 'open', 'per_page': 1}
            )
            pr_open = 0
            if pr_response and 'Link' in pr_response.headers:
                last_page = self._parse_last_page_from_link(pr_response.headers['Link'])
                if last_page:
                    pr_open = last_page
            
            pr_closed_response = self.make_request(
                f"{self.base_url}/repos/{repo_basic['full_name']}/pulls",
                {'state': 'closed', 'per_page': 1}
            )
            pr_closed = 0
            if pr_closed_response and 'Link' in pr_closed_response.headers:
                last_page = self._parse_last_page_from_link(pr_closed_response.headers['Link'])
                if last_page:
                    pr_closed = last_page
            
            # Get contributors count
            contributors_response = self.make_request(
                f"{self.base_url}/repos/{repo_basic['full_name']}/contributors",
                {'per_page': 1}
            )
            contributors = 0
            if contributors_response and 'Link' in contributors_response.headers:
                last_page = self._parse_last_page_from_link(contributors_response.headers['Link'])
                if last_page:
                    contributors = last_page
            elif contributors_response:
                try:
                    contributors = len(contributors_response.json())
                except Exception:
                    contributors = 0
            
            # Get commits count (approximate)
            commits_response = self.make_request(
                f"{self.base_url}/repos/{repo_basic['full_name']}/commits",
                {'per_page': 1}
            )
            commits = 0
            last_commit_date = ''

            if commits_response:
                try:
                    commit_items = commits_response.json()
                    if isinstance(commit_items, list) and len(commit_items) > 0:
                        iso = (
                            commit_items[0].get('commit', {}).get('committer', {}).get('date')
                            or commit_items[0].get('commit', {}).get('author', {}).get('date')
                        )
                        if iso:
                            dt = datetime.fromisoformat(iso.replace('Z', '+00:00'))
                            last_commit_date = dt.strftime('%Y-%m-%d')
                except Exception:
                    pass

                if 'Link' in commits_response.headers:
                    last_page = self._parse_last_page_from_link(commits_response.headers['Link'])
                    if last_page:
                        commits = min(last_page, 500)
                else:
                    try:
                        commits = len(commit_items) if isinstance(commit_items, list) else 0
                    except Exception:
                        commits = 0

            if not last_commit_date:
                pushed_at = repo_data.get('pushed_at')
                if pushed_at:
                    try:
                        dt = datetime.fromisoformat(pushed_at.replace('Z', '+00:00'))
                        last_commit_date = dt.strftime('%Y-%m-%d')
                    except Exception:
                        last_commit_date = ""
            
            # Get releases count
            releases_response = self.make_request(
                f"{self.base_url}/repos/{repo_basic['full_name']}/releases",
                {'per_page': 1}
            )
            releases = 0
            if releases_response and 'Link' in releases_response.headers:
                last_page = self._parse_last_page_from_link(releases_response.headers['Link'])
                if last_page:
                    releases = last_page
            elif releases_response:
                try:
                    releases = len(releases_response.json())
                except Exception:
                    releases = 0
            
            # Get README info
            readme_response = self.make_request(
                f"{self.base_url}/repos/{repo_basic['full_name']}/readme"
            )
            readme_lines = 0
            if readme_response:
                try:
                    readme_data = readme_response.json()
                    if readme_data.get('content'):
                        import base64
                        content = base64.b64decode(readme_data['content']).decode('utf-8', errors='ignore')
                        readme_lines = len(content.split('\n'))
                except Exception:
                    readme_lines = 0
            
            # Check for CI/CD (look for workflow files)
            workflow_response = self.make_request(
                f"{self.base_url}/repos/{repo_basic['full_name']}/actions/workflows"
            )
            ci_cd = False
            if workflow_response:
                try:
                    workflows = workflow_response.json()
                    ci_cd = len(workflows.get('workflows', [])) > 0
                except Exception:
                    ci_cd = False
            
            if not ci_cd:
                ci_indicators = ['.github/workflows', '.travis.yml', '.circleci', 'Jenkinsfile']
                for indicator in ci_indicators:
                    ci_response = self.make_request(
                        f"{self.base_url}/repos/{repo_basic['full_name']}/contents/{indicator}"
                    )
                    if ci_response and ci_response.status_code == 200:
                        ci_cd = True
                        break
            
            return RepoData(
                repo_name=repo_data['name'],
                url=repo_data['html_url'],
                owner=repo_data['owner']['login'],
                language=repo_data.get('language') or 'Unknown',
                license=repo_data['license']['name'] if repo_data.get('license') else 'None',
                last_commit_date=last_commit_date,
                stars=repo_data['stargazers_count'],
                forks=repo_data['forks_count'],
                watchers=repo_data['watchers_count'],
                issues_open=issues_open,
                issues_closed=issues_closed,
                pr_open=pr_open,
                pr_closed=pr_closed,
                contributors=contributors,
                commits=commits,
                releases=releases,
                readme_lines=readme_lines,
                has_wiki=repo_data.get('has_wiki', False),
                has_pages=repo_data.get('has_pages', False),
                ci_cd=ci_cd
            )
            
        except Exception as e:
            logger.error(f"Error getting detailed info for {repo_basic.get('full_name', 'unknown')}: {e}")
            return None
    
    def export_to_json(self, filename: str = '../results/github_repos.json'):
        """Export data to JSON file"""
        try:
            repos_dict_list = []
            for repo in self.repos_data:
                repo_dict = asdict(repo)
                repos_dict_list.append(repo_dict)
            
            repos_dict_list.sort(key=lambda x: (-x['stars'], x['repo_name'].lower()))
            
            for i, repo in enumerate(repos_dict_list, start=1):
                repo['index'] = i

            export_data = {
                'metadata': {
                    'total_repositories': len(repos_dict_list),
                    'export_timestamp': datetime.now().isoformat(),
                    'min_stars_threshold': self.min_stars,
                    'crawler_version': '1.1-bucketed',
                    'sorted_by': 'stars_desc, name_asc'
                },
                'repositories': repos_dict_list
            }
            
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, 'w', encoding='utf-8') as jsonfile:
                json.dump(export_data, jsonfile, indent=2, ensure_ascii=False)
                
            logger.info(f"Data exported to {filename} ({len(repos_dict_list)} repositories)")
            
            self.create_summary(repos_dict_list)
            
        except Exception as e:
            logger.error(f"Error exporting to JSON: {e}")
    
    def create_summary(self, repos_data: List[Dict]):
        """Create a summary statistics file"""
        try:
            total_repos = len(repos_data)
            if total_repos == 0:
                return
            
            languages = {}
            total_stars = sum(repo['stars'] for repo in repos_data)
            total_forks = sum(repo['forks'] for repo in repos_data)
            total_contributors = sum(repo['contributors'] for repo in repos_data)
            
            for repo in repos_data:
                lang = repo['language']
                languages[lang] = languages.get(lang, 0) + 1
            
            avg_stars = total_stars / total_repos
            avg_forks = total_forks / total_repos
            avg_contributors = total_contributors / total_repos
            
            ci_cd_repos = sum(1 for repo in repos_data if repo['ci_cd'])
            ci_cd_percentage = (ci_cd_repos / total_repos) * 100
            
            top_10_repos = sorted(repos_data, key=lambda x: x['stars'], reverse=True)[:10]
            
            summary = {
                'summary_statistics': {
                    'total_repositories': total_repos,
                    'total_stars': total_stars,
                    'total_forks': total_forks,
                    'total_contributors': total_contributors,
                    'average_stars': round(avg_stars, 2),
                    'average_forks': round(avg_forks, 2),
                    'average_contributors': round(avg_contributors, 2),
                    'repositories_with_ci_cd': ci_cd_repos,
                    'ci_cd_percentage': round(ci_cd_percentage, 2)
                },
                'language_distribution': dict(sorted(languages.items(), key=lambda x: x[1], reverse=True)),
                'top_10_repositories': [
                    {
                        'name': repo['repo_name'],
                        'owner': repo['owner'],
                        'stars': repo['stars'],
                        'language': repo['language'],
                        'url': repo['url']
                    } for repo in top_10_repos
                ],
                'generated_at': datetime.now().isoformat()
            }
            
            os.makedirs('../results', exist_ok=True)
            with open('../results/github_repos_summary.json', 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
                
            logger.info("Summary statistics saved to results/github_repos_summary.json")
            
        except Exception as e:
            logger.error(f"Error creating summary: {e}")
    
    def save_progress(self, filename: str = '../results/github_repos_progress.json'):
        """Save current progress to avoid losing data"""
        try:
            if self.repos_data:
                repos_dict_list = [asdict(repo) for repo in self.repos_data]
                progress_data = {
                    'progress_timestamp': datetime.now().isoformat(),
                    'repositories_collected': len(repos_dict_list),
                    'repositories': repos_dict_list
                }
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(progress_data, f, indent=2, ensure_ascii=False)
                logger.info(f"Progress saved: {len(repos_dict_list)} repositories")
        except Exception as e:
            logger.error(f"Error saving progress: {e}")
    
    def crawl_repositories(self, languages: List[str] = None, max_repos: int = 1000):
        """Main method to crawl repositories (NOW using bucketed search)."""
        logger.info(f"Starting to crawl repositories with minimum {self.min_stars} stars")
        
        if languages is None:
            languages = ['Python', 'JavaScript', 'Java', 'Go', 'Rust', 'TypeScript', 'C++', 'C']
        
        total_processed = 0
        seen_ids = set()
        
        try:
            for language in languages:
                if total_processed >= max_repos:
                    break
                
                logger.info(f"Crawling {language} repositories (bucketed)...")
                items = self.search_repositories_bucketing(language, self.min_stars)
                logger.info(f"Found {len(items)} {language} repositories after bucketing")
                
                for repo in items:
                    if total_processed >= max_repos:
                        break
                    if repo['id'] in seen_ids:
                        continue
                    seen_ids.add(repo['id'])

                    if repo.get('stargazers_count', 0) < self.min_stars:
                        continue
                    
                    logger.info(f"Processing: {repo['full_name']} ({repo.get('stargazers_count', 0)} stars)")
                    
                    detailed_info = self.get_detailed_repo_info(repo)
                    if detailed_info:
                        self.repos_data.append(detailed_info)
                        total_processed += 1
                        
                        if total_processed % 25 == 0:
                            self.save_progress()
                            logger.info(f"Processed {total_processed} repositories (progress saved)")
                        elif total_processed % 10 == 0:
                            logger.info(f"Processed {total_processed} repositories")
            
            logger.info(f"Crawling completed. Total repositories processed: {total_processed}")
            
        except Exception as e:
            logger.error(f"Error during crawling: {e}")
            logger.info(f"Saving progress before exit. Processed {total_processed} repositories")
        finally:
            if self.repos_data:
                self.export_to_json()


def main():
    """Main function to run the crawler"""
    tokens_env = os.environ.get("GITHUB_TOKENS", "").strip()
    tokens = [t.strip() for t in tokens_env.split(",") if t.strip()]
    
    if not tokens:
        print("Please add your GitHub personal access tokens to the script!")
        print("1. Go to https://github.com/settings/tokens")
        print("2. Generate a new token with 'repo' and 'user' scopes")
        print("3. Set env var GITHUB_TOKENS=token1,token2")
        return
    
    # Initialize crawler
    crawler = GitHubCrawler(tokens=tokens, min_stars=500)
    
    languages_to_crawl = [
        'C++', 'C',
        # Add more languages if needed
    ]
    
    try:
        crawler.crawl_repositories(
            languages=languages_to_crawl,
            max_repos=10000
        )
    except KeyboardInterrupt:
        logger.info("Crawling interrupted by user")
        if crawler.repos_data:
            logger.info("Saving collected data...")
            crawler.export_to_json()
    except Exception as e:
        logger.error(f"Error during crawling: {e}")
        if crawler.repos_data:
            logger.info("Saving collected data...")
            crawler.export_to_json()
    finally:
        logger.info("Crawler finished")


if __name__ == "__main__":
    main()
