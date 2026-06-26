"""Allow `python -m engine` to run the timeline CLI."""

from engine.run import main

if __name__ == "__main__":
    raise SystemExit(main())
