# git_crawler
Github repos crawler to crawler high star repos information. 

```shell
export GITHUB_TOKENS="token1,token2,..." 
time python repo_crawler.py
```

## Company detection

Run the repository ownership classifier from the `git_crawler` root:

```shell
export API_BASE_URL="https://your-openai-compatible-endpoint"
export API_MODEL="gpt-5.2"
export API_KEY="sk-..."
export API_PATH_TEMPLATE="/v1/chat/completions"
export API_AUTH="bearer"

python company_agent/finding_company_agent.py
```

Default files:

- Input: `data/c10k_github_repos_sorted.json`
- Output: `data/companies_repos.json`
- Checkpoint ledger: `data/companies_repos.checkpoint.jsonl`
- Run state: `data/companies_repos.state.json`

Useful options:

```shell
python company_agent/finding_company_agent.py --limit 50 --fresh
python company_agent/finding_company_agent.py --resume --max-search-results 3
python company_agent/finding_company_agent.py --resume --rerun-failure
```
