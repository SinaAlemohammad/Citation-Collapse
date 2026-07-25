import json
import os
import ssl
import time
import urllib.request
import xml.etree.ElementTree as ET
from dotenv import load_dotenv
from openai import OpenAI

# Workaround for SSL certificate verification issues on some systems
SSL_CONTEXT = ssl.create_default_context()
SSL_CONTEXT.check_hostname = False
SSL_CONTEXT.verify_mode = ssl.CERT_NONE
load_dotenv()

ARXIV_API_URL = "http://export.arxiv.org/api/query"
SEMANTIC_SCHOLAR_API_URL = "https://api.semanticscholar.org/graph/v1/paper"
SEMANTIC_SCHOLAR_DELAY_SECONDS = 5.0
SEMANTIC_SCHOLAR_MAX_RETRIES = 1
SEMANTIC_SCHOLAR_RETRY_DELAY = 10.0

ARXIV_BATCH_SIZE = 500
CUTOFF_DATE = "2022-11-01"

BUCKET_CONFIG = {
    "bucket_50_500": {
        "min_citations": 50,
        "max_citations": 500,
        "capacity": 120
    }
}

os.makedirs("buckets", exist_ok=True)
OUTPUT_FILE = "arxiv_papers.json"
PROCESSED_IDS_PATH = "processed_ids.jsonl"
CRAWLER_STATE_PATH = "crawler_state.json"


def load_buckets_from_disk():
    buckets = {name: [] for name in BUCKET_CONFIG.keys()}
    for bucket_name in BUCKET_CONFIG:
        path = f"buckets/{bucket_name}.jsonl"
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    if line.strip():
                        buckets[bucket_name].append(json.loads(line))
    return buckets


def load_seen_ids(buckets):
    seen_ids = set()
    for bucket_name, papers in buckets.items():
        for p in papers:
            seen_ids.add(p["arxiv_id"])
    if os.path.exists(PROCESSED_IDS_PATH):
        with open(PROCESSED_IDS_PATH) as f:
            for line in f:
                if line.strip():
                    seen_ids.add(line.strip())
    return seen_ids


def load_last_offset():
    if os.path.exists(CRAWLER_STATE_PATH):
        try:
            with open(CRAWLER_STATE_PATH) as f:
                data = json.load(f)
                return data.get("last_offset", 0)
        except (json.JSONDecodeError, KeyError):
            pass
    return 0


def append_processed_id(arxiv_id):
    with open(PROCESSED_IDS_PATH, "a") as f:
        f.write(arxiv_id + "\n")
        f.flush()


def search_arxiv(query, max_results=500, start=0, categories=None):
    formatted_query = f"abs:{query.replace(' ', '+AND+abs:')}"
    
    if categories:
        cat_filter = "+OR+".join(f"cat:{c}" for c in categories)
        formatted_query = f"({formatted_query})+AND+({cat_filter})"
    
    url = f"{ARXIV_API_URL}?search_query={formatted_query}&max_results={max_results}&start={start}"
    print(f"Searching arXiv (start={start}): {url}")

    with urllib.request.urlopen(url, context=SSL_CONTEXT) as response:
        return response.read().decode('utf-8')


def parse_arxiv_xml(xml_content):
    namespaces = {'atom': 'http://www.w3.org/2005/Atom'}
    root = ET.fromstring(xml_content)
    papers = []

    for entry in root.findall('atom:entry', namespaces):
        title_elem = entry.find('atom:title', namespaces)
        title = title_elem.text.strip().replace('\n', ' ') if title_elem is not None else ""

        summary_elem = entry.find('atom:summary', namespaces)
        abstract = summary_elem.text.strip().replace('\n', ' ') if summary_elem is not None else ""

        id_elem = entry.find('atom:id', namespaces)
        arxiv_url = id_elem.text if id_elem is not None else ""
        arxiv_id = arxiv_url.split('/abs/')[-1].split('v')[0] if arxiv_url else ""

        pdf_url = ""
        for link in entry.findall('atom:link', namespaces):
            if link.get('title') == 'pdf':
                pdf_url = link.get('href', '')
                break

        authors = []
        for author in entry.findall('atom:author', namespaces):
            name_elem = author.find('atom:name', namespaces)
            if name_elem is not None and name_elem.text:
                authors.append(name_elem.text.strip())

        published_elem = entry.find('atom:published', namespaces)
        published_date = published_elem.text.strip() if published_elem is not None else ""

        if not (title and abstract and arxiv_id):
            continue

        # ISO dates sort lexicographically, so string comparison works
        if published_date >= CUTOFF_DATE:
            continue

        papers.append({
            'title': title,
            'abstract': abstract,
            'arxiv_id': arxiv_id,
            'pdf_url': pdf_url,
            'authors': authors,
            'published_date': published_date,
        })

    print(f"Parsed {len(papers)} pre-{CUTOFF_DATE} papers from arXiv response")
    return papers


def verify_with_llm(paper, client):
    prompt = f"""Determine if this academic paper is specifically about knowledge distillation or model compression in deep learning or NLP.

        Abstract: {paper['abstract']}

        Answer with only 'yes' or 'no'. The paper should be about transferring knowledge from a larger to a smaller model as a core topic — this includes knowledge distillation, relation distillation, task-specific distillation, model compression, or teacher-student training. 
        Exclude papers that only mention these concepts casually.
        """ 

    response = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that classifies academic papers. Respond only with 'yes' or 'no'."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_completion_tokens=200
    )

    answer = response.choices[0].message.content.strip().lower()
    is_relevant = 'yes' in answer
    return is_relevant


def get_citation_count(arxiv_id):
    url = f"{SEMANTIC_SCHOLAR_API_URL}/arXiv:{arxiv_id}?fields=citationCount"
    request = urllib.request.Request(
        url,
        headers={'User-Agent': 'ArxivPaperCollector/1.0'}
    )

    last_exception = None

    for attempt in range(SEMANTIC_SCHOLAR_MAX_RETRIES + 1):
        # Throttle to avoid 429s (~100 req / 5 min unauthenticated)
        time.sleep(SEMANTIC_SCHOLAR_DELAY_SECONDS)

        try:
            with urllib.request.urlopen(request, context=SSL_CONTEXT) as response:
                data = json.loads(response.read().decode('utf-8'))

            citation_count = data.get('citationCount', 0)
            if citation_count is None:
                citation_count = 0

            return citation_count

        except Exception as e:
            last_exception = e

            if attempt < SEMANTIC_SCHOLAR_MAX_RETRIES:
                # Exponential backoff: 10s, 20s, 40s...
                retry_delay = SEMANTIC_SCHOLAR_RETRY_DELAY * (2 ** attempt)
                print(f"  Retry {attempt + 1}/{SEMANTIC_SCHOLAR_MAX_RETRIES}: {e}")
                time.sleep(retry_delay)

    raise last_exception


def get_bucket_name(citation_count):
    for bucket_name, config in BUCKET_CONFIG.items():
        if not config['max_citations'] and citation_count >= config['min_citations']:
            return bucket_name

        if config['min_citations'] <= citation_count <= config['max_citations']:
            return bucket_name

    return None


def add_to_bucket(paper, citation_count, buckets):
    bucket_name = get_bucket_name(citation_count)
    bucket_capacity = BUCKET_CONFIG[bucket_name]['capacity']

    if len(buckets[bucket_name]) >= bucket_capacity:
        return False

    paper_with_citations = {
        'title': paper['title'],
        'abstract': paper['abstract'],
        'arxiv_id': paper['arxiv_id'],
        'citation_count': citation_count,
        'url': paper['pdf_url'],
        'authors': paper['authors'],
        'published_date': paper['published_date'],
    }

    buckets[bucket_name].append(paper_with_citations)
    count = len(buckets[bucket_name])
    print(f"  Added to '{bucket_name}' ({count}/{bucket_capacity}): {paper['title']}")

    with open(f"buckets/{get_bucket_name(citation_count)}.jsonl", 'a') as f:
        f.write(json.dumps(paper_with_citations) + '\n')
        f.flush()

    return True


def buckets_are_full(buckets):
    for bucket_name, config in BUCKET_CONFIG.items():
        if len(buckets[bucket_name]) < config['capacity']:
            return False
    return True


def print_summary(buckets):
    total = 0
    for bucket_name, papers in buckets.items():
        capacity = BUCKET_CONFIG[bucket_name]['capacity']
        count = len(papers)
        total += count
        print(f"  {bucket_name}: {count}/{capacity} papers")
    print(f"Total papers collected: {total}")


def process_papers(papers, buckets, client, start_index=0):

    for i, paper in enumerate(papers):
        global_index = start_index + i + 1
        print(f"\n[Paper {global_index}] {paper['title']}")

        if buckets_are_full(buckets):
            print("All buckets are full!")
            return True

        try:
            citation_count = get_citation_count(paper['arxiv_id'])
            print(f"  Citations: {citation_count}")
        except Exception as e:
            print(f"  Skipping: Could not get citation count - {e}")
            continue

        bucket_name = get_bucket_name(citation_count)
        if not bucket_name:
            append_processed_id(paper['arxiv_id'])
            continue

        bucket_capacity = BUCKET_CONFIG[bucket_name]['capacity']
        if len(buckets[bucket_name]) >= bucket_capacity:
            print(f"  Skipping: Bucket '{bucket_name}' already full")
            append_processed_id(paper['arxiv_id'])
            continue

        try:
            is_relevant = verify_with_llm(paper, client)
            if not is_relevant:
                print(f"  [LLM: no]  {paper['title']}")
                append_processed_id(paper['arxiv_id'])
                continue
            print(f"  [LLM: yes] {paper['title']}")
        except Exception as e:
            print(f"  LLM verification failed: {e}")
            append_processed_id(paper['arxiv_id'])
            continue

        add_to_bucket(paper, citation_count, buckets)
        append_processed_id(paper['arxiv_id'])

    return buckets_are_full(buckets)


def main():

    openai_api_key = os.getenv('OPENAI_API_KEY')
    if not openai_api_key:
        print("Error: OPENAI_API_KEY not found in environment")
        return

    client = OpenAI(api_key=openai_api_key)
    buckets = load_buckets_from_disk()
    seen_ids = load_seen_ids(buckets)
    start_offset = load_last_offset()
    initial_seen_count = len(seen_ids)

    total_papers_processed = initial_seen_count
    batch_number = start_offset // ARXIV_BATCH_SIZE

    if buckets_are_full(buckets):
        print("All buckets already full. Nothing to do.")
        print_summary(buckets)
        return

    print(f"Resuming: {sum(len(p) for p in buckets.values())} papers in buckets, "
          f"{len(seen_ids)} seen IDs, starting arXiv from offset {start_offset}")

    while not buckets_are_full(buckets):
        batch_number += 1
        print(f"\n[Batch {batch_number}] Fetching papers {start_offset + 1} to {start_offset + ARXIV_BATCH_SIZE}")

        try:
            xml_content = search_arxiv(
                "knowledge distillation",
                max_results=ARXIV_BATCH_SIZE,
                start=start_offset,
                categories=["cs.CL", "cs.LG", "cs.AI"]
            )
            papers = parse_arxiv_xml(xml_content)
        except Exception as e:
            print(f"Error searching arXiv: {e}")
            break

        if not papers:
            print("No more papers found from arXiv")
            break

        papers = [p for p in papers if p['arxiv_id'] not in seen_ids]
        for p in papers:
            seen_ids.add(p['arxiv_id'])

        if not papers:
            print("No new papers in this batch (all duplicates)")
            start_offset += ARXIV_BATCH_SIZE
            with open(CRAWLER_STATE_PATH, "w") as f:
                json.dump({"last_offset": start_offset}, f)
            continue

        print(f"Processing {len(papers)} papers from this batch...")

        all_full = process_papers(papers, buckets, client, start_index=total_papers_processed)
        total_papers_processed += len(papers)

        start_offset += ARXIV_BATCH_SIZE
        with open(CRAWLER_STATE_PATH, "w") as f:
            json.dump({"last_offset": start_offset}, f)

        if all_full:
            break

        print_summary(buckets)

    print_summary(buckets)
    print(f"Total papers processed this run: {total_papers_processed - initial_seen_count}")
    print(f"Batches fetched: {batch_number}")


if __name__ == "__main__":
    main()
