"""Position reconciliation — compares broker, persisted, and internal state on startup."""

import asyncio
from typing import Dict, List, Optional
from loguru import logger


class PositionReconciler:
    """
    Reconciles positions across three sources on startup:
    1. Broker (actual open positions from API)
    2. Persisted (state saved to disk from last run)
    3. Internal (ExecutionEngine in-memory tracking)

    Resolves discrepancies and produces a unified view.
    """

    def __init__(self, execution_engine, persistence):
        self._engine = execution_engine
        self._persistence = persistence

    async def reconcile(self) -> List[Dict]:
        """Run reconciliation. Returns list of resolved positions."""
        broker_positions = await self._fetch_broker_positions()
        persisted_positions = self._persistence.load_positions()
        internal_positions = self._get_internal_positions()

        resolved = []
        issues = []

        broker_by_id = {p.get("position_id", ""): p for p in broker_positions}
        persisted_by_id = {p.get("position_id", ""): p for p in persisted_positions}

        all_ids = set(broker_by_id.keys()) | set(persisted_by_id.keys())

        for pid in all_ids:
            in_broker = pid in broker_by_id
            in_persisted = pid in persisted_by_id
            in_internal = pid in {str(t.position_id) for t in self._engine.open_positions.values()}

            if in_broker and not in_internal:
                pos = broker_by_id[pid]
                if in_persisted:
                    issues.append(f"Position {pid}: on broker + persisted but not in engine — re-adding")
                    self._engine.open_positions[int(pid)] = self._persistence._to_trade_record(pos)
                else:
                    issues.append(f"Position {pid}: on broker only — tracked from broker state")
                    self._engine.open_positions[int(pid)] = self._persistence._to_trade_record(pos)
                resolved.append(pos)

            elif in_persisted and not in_broker:
                pos = persisted_by_id[pid]
                issues.append(f"Position {pid}: persisted but not on broker — removed from tracking")
                if in_internal:
                    del self._engine.open_positions[int(pid)]
                resolved.append(pos)

            elif not in_broker and not in_persisted and in_internal:
                pos = next(
                    t for t in self._engine.open_positions.values()
                    if str(t.position_id) == pid
                )
                issues.append(f"Position {pid}: internal only — closed on broker, recording manually")
                self._engine._finalize_close(
                    pos, pos.entry_price, 0.0, "reconciled_closed_on_broker"
                )
                resolved.append({"position_id": pid, "status": "closed"})

            else:
                if in_broker:
                    resolved.append(broker_by_id[pid])

        for issue in issues:
            logger.warning(f"[reconcile] {issue}")

        logger.info(
            f"[reconcile] {len(broker_positions)} broker / {len(persisted_positions)} persisted "
            f"/ {len(internal_positions)} internal → {len(resolved)} resolved"
        )
        return resolved

    async def _fetch_broker_positions(self) -> List[Dict]:
        try:
            if hasattr(self._engine.client, "get_open_positions"):
                return self._engine.client.get_open_positions()
        except Exception as e:
            logger.warning(f"[reconcile] Could not fetch broker positions: {e}")
        return []

    def _get_internal_positions(self) -> List[Dict]:
        return list(self._engine.open_positions.values())

    async def reconcile_account_balance(self, risk_manager) -> bool:
        """Sync risk manager's balance with broker. Returns True if synced."""
        try:
            info = await self._fetch_account_info()
            if info and info.get("balance", 0) > 0:
                risk_manager.initial_balance = info["balance"]
                risk_manager.peak_balance = max(
                    risk_manager.peak_balance, info["balance"]
                )
                risk_manager.kill_switch_triggered = False
                logger.info(f"[reconcile] Balance synced: ${info['balance']:,.2f}")
                return True
        except Exception as e:
            logger.warning(f"[reconcile] Balance sync failed: {e}")
        return False

    async def _fetch_account_info(self) -> Optional[Dict]:
        try:
            acc = self._engine.get_account_info()
            if asyncio.iscoroutine(acc):
                return await acc
            return acc
        except Exception:
            return None
