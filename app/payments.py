from __future__ import annotations

from decimal import Decimal
from typing import Any


class PaymentsClient:
    def refund_preview(
        self,
        *,
        thread_id: str,
        issue_type: str,
        recommendation: str,
        order_id: str | None = None,
    ) -> dict[str, Any]:
        amount_map = {
            "duplicate_charge": Decimal("129.99"),
            "late_delivery": Decimal("15.00"),
            "missing_item": Decimal("20.00"),
            "refund_request": Decimal("89.99"),
            "warranty": Decimal("0.00"),
        }
        action = "refund_preview" if issue_type != "warranty" else "replacement_preview"
        return {
            "preview_id": f"preview-{thread_id}",
            "thread_id": thread_id,
            "order_id": order_id,
            "issue_type": issue_type,
            "action": action,
            "amount": float(amount_map.get(issue_type, Decimal("0.00"))),
            "currency": "USD",
            "reason": recommendation,
        }

    def refund_commit(self, preview: dict[str, Any]) -> dict[str, Any]:
        return {
            "commit_id": f"commit-{preview['thread_id']}",
            "preview_id": preview["preview_id"],
            "order_id": preview.get("order_id"),
            "status": "committed",
        }
