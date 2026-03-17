import os
import time
import requests
import xml.etree.ElementTree as ET
from urllib.parse import quote
from collections import Counter

S2_API_BASE = "https://api.semanticscholar.org/graph/v1"


def _s2_request(url, params=None):
    """Make S2 API request. Uses S2_API_KEY if set for higher rate limits."""
    headers = {}
    if "S2_API_KEY" in os.environ:
        headers["x-api-key"] = os.environ["S2_API_KEY"]
    r = requests.get(url, headers=headers or None, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def get_s2_citations_by_year(paper_id: str):
    """
    Get citation count and citations by year from Semantic Scholar API.

    Args:
        paper_id: S2 paper ID, e.g. "CorpusId:12345" or "arXiv:1510.00087"

    Returns:
        (total_citation_count, counts_by_year) where counts_by_year is
        [{"year": 2020, "cited_by_count": 5}, ...] sorted by year.
    """
    url = f"{S2_API_BASE}/paper/{paper_id}/citations"
    year_counts = Counter()
    offset = 0
    limit = 1000

    while True:
        params = {"fields": "citingPaper.year", "limit": limit, "offset": offset}
        data = _s2_request(url, params)
        items = data.get("data", [])
        for item in items:
            citing = item.get("citingPaper") or {}
            year = citing.get("year")
            if year is not None:
                year_counts[int(year)] += 1

        if "next" not in data:
            break
        offset = data["next"]
        time.sleep(0.5)  # rate limit

    total = sum(year_counts.values())
    counts_by_year = [{"year": y, "cited_by_count": c} for y, c in sorted(year_counts.items())]
    return total, counts_by_year


def get_s2_paper(paper_id: str, fields=None):
    """Fetch paper metadata from S2. paper_id: CorpusId:123 or arXiv:1510.00087"""
    if fields is None:
        fields = ["paperId", "title", "year", "citationCount"]
    params = {"fields": ",".join(fields)}
    url = f"{S2_API_BASE}/paper/{paper_id}"
    return _s2_request(url, params)


def get_citation_s2(paper_id: str):
    """
    Get citation count by year from Semantic Scholar.

    Args:
        paper_id: "CorpusId:12345" or "arXiv:1510.00087"

    Returns:
        (total, counts_by_year, publication_year)

    Note: Set S2_API_KEY env var for higher rate limits (100 req/5min without key).
    """
    paper = get_s2_paper(paper_id, fields=["year"])
    pub_year = paper.get("year")
    total, counts_by_year = get_s2_citations_by_year(paper_id)
    return total, counts_by_year, pub_year


def counts_by_year_to_trajectory(counts_by_year: list, pub_year: int) -> list:
    """
    Convert counts_by_year to cumulative citation_trajectory (list of ints).
    trajectory[i] = cumulative citations through year (pub_year + i).

    Args:
        counts_by_year: [{"year": 2020, "cited_by_count": 5}, ...]
        pub_year: Publication year

    Returns:
        [c0, c1, c2, ...] where c0 = citations in pub_year, c1 = cumulative through pub_year+1, etc.
    """
    if not counts_by_year or pub_year is None:
        return []
    year2count = {x["year"]: x["cited_by_count"] for x in counts_by_year}
    max_year = max(year2count.keys())
    trajectory = []
    cum = 0
    for y in range(pub_year, max_year + 1):
        cum += year2count.get(y, 0)
        trajectory.append(int(cum))
    return trajectory


def get_citation_s2_batch(paper_ids: list, sleep_between=1.0):
    """
    Get citation-by-year for multiple papers. Yields (paper_id, total, counts_by_year, pub_year).

    Args:
        paper_ids: List of S2 paper IDs (CorpusId, arXiv ID, DOI, or paperId hash)
        sleep_between: Seconds to sleep between requests (rate limit)
    """
    for pid in paper_ids:
        sid = str(pid)
        if sid.startswith(("CorpusId:", "arXiv:", "DOI:")):
            api_id = sid
        elif sid.isdigit():
            api_id = f"CorpusId:{sid}"
        else:
            api_id = sid  # S2 paperId hash
        try:
            total, by_year, pub_year = get_citation_s2(api_id)
            yield pid, total, by_year, pub_year
        except Exception:
            yield pid, None, [], None
        time.sleep(sleep_between)


def get_arxiv_metadata(arxiv_id: str):
    url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"
    r = requests.get(url, timeout=30)
    r.raise_for_status()

    root = ET.fromstring(r.text)
    ns = {
        "atom": "http://www.w3.org/2005/Atom",
        "arxiv": "http://arxiv.org/schemas/atom",
    }

    entry = root.find("atom:entry", ns)
    if entry is None:
        raise ValueError(f"No arXiv record found for {arxiv_id}")

    title = entry.findtext("atom:title", default="", namespaces=ns).strip()
    published = entry.findtext("atom:published", default="", namespaces=ns).strip()
    doi = entry.findtext("arxiv:doi", default=None, namespaces=ns)

    return {
        "title": title,
        "published": published,
        "doi": doi,
    }

def get_openalex_by_doi(doi: str):
    doi = doi.lower().removeprefix("https://doi.org/").removeprefix("doi:")
    url = f"https://api.openalex.org/works/https://doi.org/{doi}"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.json()

def trailing_3_year_citations(counts_by_year, current_year=None):
    from datetime import datetime
    if current_year is None:
        current_year = datetime.utcnow().year
    years = {current_year - 1, current_year - 2, current_year - 3}
    return sum(x["cited_by_count"] for x in counts_by_year if x["year"] in years)

def first_3_year_citations(counts_by_year, publication_year):
    years = {publication_year, publication_year + 1, publication_year + 2}
    return sum(x["cited_by_count"] for x in counts_by_year if x["year"] in years)


def search_openalex_by_title(title: str):
    url = f"https://api.openalex.org/works?search={quote(title)}&per-page=5"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.json()["results"]

def get_citation(arxiv_id):
    meta = get_arxiv_metadata(arxiv_id)
    if meta["doi"]:
        work = get_openalex_by_doi(meta["doi"])
        total = work["cited_by_count"]
        by_year = work.get("counts_by_year", [])
        pub_year = int(meta["published"][:4])
        return total, by_year, pub_year
    else:
        results = search_openalex_by_title(meta["title"])
        for w in results:
            if w["display_name"] == meta["title"]:
                print("Title:", w["display_name"], "Publication year:", w["publication_year"], "Cited by count:", w["cited_by_count"], "ID:", w["id"])
                return w["cited_by_count"], w.get("counts_by_year", []), int(w["publication_year"])
        return None, None, None


if __name__ == "__main__":
    # Example: Semantic Scholar by corpus_id or arXiv ID
    for pid in ["ffe9482f111067c5a76703463624dbe885f966a3"]:
        print(f"\n--- {pid} ---")
        try:
            total, by_year, pub_year = get_citation_s2(pid)
            print(f"Total citations: {total}, publication year: {pub_year}")
            print("Citations by year:", by_year)
        except Exception as e:
            raise Exception(f"Error: {e}")
