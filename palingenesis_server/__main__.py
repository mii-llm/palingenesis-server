import argparse

import uvicorn

from .api import create_app


def main():
    parser = argparse.ArgumentParser(description="Serve Palingenesis jobs on this host")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run(create_app(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
