from __future__ import annotations

import argparse
import time

from data_fetch.fetch_from_phoenix import (
    build_default_adapter,
    fetch_phoenix_traces_and_spans,
    make_fetch_request,
)

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

    fetch_parser = subparsers.add_parser("fetch-phoenix", help="Fetch traces/spans from Phoenix")
    fetch_parser.add_argument("--from", dest="from_time", help="Start time (ISO-8601, UTC preferred)")
    fetch_parser.add_argument("--to", dest="to_time", help="End time (ISO-8601). Defaults to now UTC")
    fetch_parser.add_argument("--delta", action="store_true", default=False)
    fetch_parser.add_argument("--project-name", dest="project_name")
    fetch_parser.add_argument("--span-kind", dest="span_kind")
    fetch_parser.add_argument("--batch-size", dest="batch_size", type=int)
    fetch_parser.add_argument("--out", dest="output_dir")
    fetch_parser.add_argument("--checkpoint-file", dest="checkpoint_file")
    fetch_parser.add_argument("--no-update-checkpoint", action="store_true", default=False)
    fetch_parser.add_argument(
        "--poll-interval-seconds",
        dest="poll_interval_seconds",
        type=int,
        default=0,
        help="Optional loop mode. If > 0, reruns fetch on this interval.",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    settings = get_settings()

    if args.command == "serve":
        host = args.host if args.host != "127.0.0.1" else settings.server_host
        port = args.port if args.port != 8000 else settings.server_port
        print(f"Starting RAG Bot API server on http://{host}:{port}")
        uvicorn.run("ragbot.api.app:app", host=host, port=port, reload=settings.server_reload)
        return
    
    if args.command == "fetch-phoenix":
        adapter = build_default_adapter(settings.phoenix_query_endpoint)

        request = make_fetch_request(
            from_time=args.from_time,
            to_time=args.to_time,
            project_name=args.project_name or settings.phoenix_project_name,
            span_kind=args.span_kind or settings.phoenix_fetch_span_kind,
            batch_size=args.batch_size or settings.phoenix_fetch_batch_size,
            output_dir=args.output_dir or settings.phoenix_fetch_output_dir,
            checkpoint_file=args.checkpoint_file or settings.phoenix_fetch_checkpoint_file,
            delta=args.delta,
            update_checkpoint=not args.no_update_checkpoint,
        )

        interval = max(args.poll_interval_seconds or 0, 0)
        if interval == 0:
            summary = fetch_phoenix_traces_and_spans(request, adapter)
            print(summary.model_dump_json(indent=2))
            return

        while True:
            summary = fetch_phoenix_traces_and_spans(request, adapter)
            print(summary.model_dump_json(indent=2))
            time.sleep(interval)

    service = ChatService.create(settings)

    if args.command == "ingest":
        result = service.ingestion.ingest_file(args.source, rebuild=args.rebuild)
        print(result.model_dump_json(indent=2))
        return


if __name__ == "__main__":
    main()
