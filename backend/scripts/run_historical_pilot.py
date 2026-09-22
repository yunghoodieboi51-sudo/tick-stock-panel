"""Run the explicitly bounded V4.7c historical-data pilot.

This script intentionally has no scheduler/API entrypoint.  ``--live`` is
required because it writes canonical parquet after creating a local backup.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from app.config import settings
from app.data_providers import get_provider
from app.services.historical_pilot import run_pilot
from app.tickflow.repository import DataStore, KlineRepository


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="allow canonical pilot writes")
    parser.add_argument("--force", action="store_true", help="retry completed batches")
    parser.add_argument("--runtime-dir", type=Path, default=None)
    args = parser.parse_args()
    if not args.live:
        parser.error("--live is required; this bounded pilot writes canonical parquet")
    data_dir = Path(settings.data_dir)
    runtime = args.runtime_dir or data_dir / "research_manifests"
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    repo = KlineRepository(DataStore(data_dir))
    provider = get_provider("tickflow")
    result = run_pilot(
        repo,
        provider,
        manifest_path=runtime / "research_daily_v1_pilot.manifest.json",
        report_path=runtime / "research_daily_v1_pilot.coverage.json",
        backup_root=data_dir / ".pilot_backups" / f"research_daily_v1_pilot_{stamp}",
        force=args.force,
    )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
