"""Allow ``python -m tensorcad_runtime ...`` alongside the ``tensorcad-runtime`` script."""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
