"""
FormaCore AI - executor/versioning.py
Board state history with undo/redo. Stores snapshots in memory.
Each confirmed action creates a new version.
"""
from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime
from typing import Dict, Any, List, Optional

from core.grid import Grid, Net


class BoardVersion:
    """A single snapshot in the version history."""

    def __init__(self, grid: Grid, nets: List[Net],
                 action_id: str, payload: Dict[str, Any],
                 result: Dict[str, Any]):
        self.grid = grid
        self.nets = nets
        self.action_id = action_id
        self.payload = payload
        self.result = result
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class VersionManager:
    """
    Manages board state versions. Stores grid snapshots in memory.
    Supports undo/redo and browsing history.
    """

    def __init__(self, initial_grid: Grid, nets: List[Net],
                 max_history: int = 50):
        self.nets = nets
        self.max_history = max_history
        self._history: List[BoardVersion] = []
        self._undo_stack: List[BoardVersion] = []

        # Store initial state as version 0
        initial = BoardVersion(
            grid=initial_grid.clone(),
            nets=nets,
            action_id="initial",
            payload={},
            result={"status": "ok", "msg": "Initial board state"},
        )
        initial.grid.components = [deepcopy(c)
                                   for c in initial_grid.components]
        initial.grid.routed_paths = {
            k: list(v) for k, v in initial_grid.routed_paths.items()
        }
        self._history.append(initial)

    @property
    def current_grid(self) -> Grid:
        """Return the latest board state (clone for safety)."""
        g = self._history[-1].grid.clone()
        g.components = [deepcopy(c)
                        for c in self._history[-1].grid.components]
        g.routed_paths = {
            k: list(v)
            for k, v in self._history[-1].grid.routed_paths.items()
        }
        return g

    @property
    def version_count(self) -> int:
        return len(self._history)

    def commit(self, grid: Grid, action_id: str,
               payload: Dict[str, Any],
               result: Dict[str, Any]) -> int:
        """
        Save a new version after a confirmed action.
        Returns the new version number.
        """
        snapshot = grid.clone()
        snapshot.components = [deepcopy(c) for c in grid.components]
        snapshot.routed_paths = {k: list(v) for k, v in grid.routed_paths.items()}

        version = BoardVersion(
            grid=snapshot,
            nets=self.nets,
            action_id=action_id,
            payload=payload,
            result=result,
        )
        self._history.append(version)
        self._undo_stack.clear()  # new commit clears redo stack

        # Trim old versions if exceeding max
        if len(self._history) > self.max_history:
            self._history = self._history[-self.max_history:]

        return len(self._history) - 1

    def undo(self) -> Optional[Grid]:
        """
        Undo the last action. Returns the restored grid, or None if
        already at initial state.
        """
        if len(self._history) <= 1:
            return None  # can't undo past initial state

        removed = self._history.pop()
        self._undo_stack.append(removed)

        g = self._history[-1].grid.clone()
        g.components = [deepcopy(c)
                        for c in self._history[-1].grid.components]
        g.routed_paths = {
            k: list(v)
            for k, v in self._history[-1].grid.routed_paths.items()
        }
        return g

    def redo(self) -> Optional[Grid]:
        """
        Redo a previously undone action. Returns the restored grid,
        or None if nothing to redo.
        """
        if not self._undo_stack:
            return None

        version = self._undo_stack.pop()
        self._history.append(version)

        g = version.grid.clone()
        g.components = [deepcopy(c) for c in version.grid.components]
        g.routed_paths = {
            k: list(v) for k, v in version.grid.routed_paths.items()
        }
        return g

    def get_history(self) -> List[Dict[str, Any]]:
        """Return a summary of all versions for display."""
        entries = []
        for i, v in enumerate(self._history):
            entries.append({
                "version": i,
                "timestamp": v.timestamp,
                "action": v.action_id,
                "payload": v.payload,
                "status": v.result.get("status", "unknown"),
            })
        return entries

    def get_version(self, index: int) -> Optional[Grid]:
        """Get a specific version's grid state."""
        if 0 <= index < len(self._history):
            g = self._history[index].grid.clone()
            g.components = [deepcopy(c)
                            for c in self._history[index].grid.components]
            g.routed_paths = {
                k: list(v)
                for k, v in self._history[index].grid.routed_paths.items()
            }
            return g
        return None

    def export_log(self) -> str:
        """Export version history as JSON string."""
        log = []
        for i, v in enumerate(self._history):
            log.append({
                "version": i,
                "timestamp": v.timestamp,
                "action": v.action_id,
                "payload": _safe_payload(v.payload),
                "result_status": v.result.get("status", "unknown"),
            })
        return json.dumps(log, indent=2)


def _safe_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Make payload JSON-serializable."""
    safe = {}
    for k, v in payload.items():
        if isinstance(v, (str, int, float, bool, type(None))):
            safe[k] = v
        elif isinstance(v, (list, tuple)):
            safe[k] = list(v)
        elif isinstance(v, dict):
            safe[k] = _safe_payload(v)
        else:
            safe[k] = str(v)
    return safe
