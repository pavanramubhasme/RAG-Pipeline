"""Command line.

python -m ragapp.cli warmup                 # download the open-source models
python -m ragapp.cli ingest data/sample     # index a folder or files
python -m ragapp.cli ask "How many vacation days do I get?"
python -m ragapp.cli ask "What is EMP-7001's salary?" --user finance_demo   # as a demo user
python -m ragapp.cli graph                  # print the LangGraph workflow as Mermaid

The CLI opens the index files directly, so by default it runs as the local operator with
full access (anyone who can read the files can read the data). Access control protects
the API; use --user to see what a given role would get.
"""

import argparse
import json

from ragapp.access import UNRESTRICTED
from ragapp.config import get_settings
from ragapp.observability import configure_logging, setup_tracing, shutdown_tracing


def main() -> None:
    parser = argparse.ArgumentParser(prog="ragapp")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("warmup", help="download and cache embedding/BM25/reranker models")
    ingest = sub.add_parser("ingest", help="index files or folders")
    ingest.add_argument("paths", nargs="+")
    ask = sub.add_parser("ask", help="ask a question")
    ask.add_argument("question")
    ask.add_argument("--user", help="answer as this user from the identities file (e.g. hr_demo)")
    sub.add_parser("graph", help="print the workflow diagram (Mermaid)")
    args = parser.parse_args()

    settings = get_settings()
    configure_logging("WARNING")

    if args.command == "warmup":
        from ragapp.factory import build_encoders

        dense, sparse, _ = build_encoders(settings)
        dense.embed_query("warm up")
        sparse.embed_query("warm up")
        print("Models cached.")
        return

    from ragapp.factory import build_service

    setup_tracing(settings)
    service = build_service(settings)
    try:
        if args.command == "ingest":
            print(service.ingest(args.paths).model_dump_json(indent=2))
        elif args.command == "graph":
            print(service.graph.get_graph().draw_mermaid())
        elif args.command == "ask":
            if args.user and service.access is None:
                parser.error("--user needs access control on (RAG_ACCESS_CONTROL=true)")
            identity = service.access.user(args.user) if args.user else UNRESTRICTED
            r = service.ask(args.question, identity=identity)
            print(f"\n{r.answer}\n")
            for c in r.citations:
                label = f" [{c.classification}]" if c.classification else ""
                print(f"  [{c.ref}] {c.source} | {c.section or '-'}{label}  (relevance {c.score or 0:.2f})")
            if r.access:
                print(f"\nas: {r.access.user} ({r.access.role}), may see: {', '.join(r.access.scopes)}")
            print(f"\npath: {' -> '.join(r.steps)}   grounded: {r.grounded}")
            if r.search_query:
                print(f"rewritten query: {r.search_query}")
            print(f"timings: {json.dumps(r.timings_ms)}")
    finally:
        service.close()
        shutdown_tracing()


if __name__ == "__main__":
    main()
