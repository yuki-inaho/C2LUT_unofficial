"""Support ``python -m c2lut`` without special working-directory requirements."""

from .cli import main

raise SystemExit(main())
