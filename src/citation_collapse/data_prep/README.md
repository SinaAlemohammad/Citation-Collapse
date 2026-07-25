# data_prep — build the seed from scratch (optional)

`crawler.py` regenerates the raw seed corpus. You do **not** need it to run anything —
`data/seed.jsonl` already ships. Use it only to build a fresh seed on a new topic.

It crawls arXiv, uses an LLM (gpt-5-mini) to verify each paper is on-topic, buckets by
citation count, and writes `data/buckets/bucket_50_500.jsonl`. That bucket is then
standardized into `data/seed.jsonl` (ids, fake author/year, title, abstract).

Standalone: imports only `openai`, `python-dotenv`, and stdlib. Needs `OPENAI_API_KEY`.
