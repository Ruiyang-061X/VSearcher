import requests
import argparse

from data_synthesis_util import *


WDQS_URL = "https://query.wikidata.org/sparql"
HEADERS = {
    "Accept": "application/sparql-results+json",
    "User-Agent": "less-known-entities/1.0 (you@example.com)",
}

def fetch_wikidata_candidates_one_shot(limit, sitelinks_max, min_statements):
    query = f"""
    SELECT ?item ?title WHERE {{
      ?article schema:about ?item ;
               schema:isPartOf <https://en.wikipedia.org/> ;
               schema:name ?title .
      ?item wikibase:sitelinks ?sitelinks .
      ?item wikibase:statements ?statements .
      ?item wdt:P18 ?image .
      FILTER(?sitelinks <= {sitelinks_max})
      FILTER(?statements >= {min_statements})
      MINUS {{ ?item wdt:P31 wd:Q5 . }}
    }}
    LIMIT {limit}
    """

    r = requests.get(WDQS_URL, params={"query": query}, headers=HEADERS, timeout=90)
    r.raise_for_status()
    rows = r.json()["results"]["bindings"]

    out = []
    for b in rows:
        qid = b["item"]["value"].rsplit("/", 1)[-1]
        out.append({
            "qid": qid,
            "title": b["title"]["value"],
        })
    return out

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=150000)
    parser.add_argument("--sitelinks_max", type=int, default=10)
    parser.add_argument("--min_statements", type=int, default=20)
    parser.add_argument("--result_path", type=str, required=True)
    args = parser.parse_args()

    seed_list = fetch_wikidata_candidates_one_shot(args.limit, args.sitelinks_max, args.min_statements)
    save_jsonl(seed_list, args.result_path)
    print(f"Saved to {args.result_path}")