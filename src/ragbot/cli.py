from __future__ import annotations

import argparse

import uvicorn

from ragbot.config import get_settings
from ragbot.service import ChatService


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ragbot")
    subparsers = parser.add_subparsers(dest="command", required=True)

    serve_parser = subparsers.add_parser("serve", help="Run the FastAPI service")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8000)

    ingest_parser = subparsers.add_parser("ingest", help="Ingest a text file into FAISS")
    ingest_parser.add_argument("--source", required=True)
    ingest_parser.add_argument("--rebuild", action="store_true", default=False)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    service = ChatService.create(get_settings())

    if args.command == "serve":
        uvicorn.run("ragbot.api.app:app", host=args.host, port=args.port, reload=False)
        return

    if args.command == "ingest":
        result = service.ingestion.ingest_file(args.source, rebuild=args.rebuild)
        print(result.model_dump_json(indent=2))
        return


if __name__ == "__main__":
    main()
