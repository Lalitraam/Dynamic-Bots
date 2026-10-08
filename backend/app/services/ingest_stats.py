"""
Ingestion statistics tracking data structure.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


@dataclass
class IngestStats:
    raw_lines: int = 0
    requests: int = 0
    retries_used: int = 0
    skipped_duplicates: int = 0
    request_log: list[dict] = field(default_factory=list)
    bucket_requests: dict[str, int] = field(default_factory=dict)
    drop_stats: dict[str, int] = field(default_factory=dict)
    quota_used: dict[str, int] = field(default_factory=dict)
    bucket_counts: dict[str, int] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return {
            "raw_lines": self.raw_lines,
            "requests": self.requests,
            "retries_used": self.retries_used,
            "skipped_duplicates": self.skipped_duplicates,
            "request_log": self.request_log,
            "bucket_requests": self.bucket_requests,
            "drop_stats": self.drop_stats,
            "quota_used": self.quota_used,
            "bucket_counts": self.bucket_counts,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "IngestStats":
        return cls(
            raw_lines=data.get("raw_lines", 0),
            requests=data.get("requests", 0),
            retries_used=data.get("retries_used", 0),
            skipped_duplicates=data.get("skipped_duplicates", 0),
            request_log=data.get("request_log", []),
            bucket_requests=data.get("bucket_requests", {}),
            drop_stats=data.get("drop_stats", {}),
            quota_used=data.get("quota_used", {}),
            bucket_counts=data.get("bucket_counts", {}),
            created_at=data.get("created_at", ""),
        )

