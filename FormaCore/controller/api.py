"""
FormaCore AI - controller/api.py
Orchestrates the full action pipeline:
  validate -> sandbox preview -> apply (on confirm).

Called directly by the Streamlit UI. No network layer needed.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Dict, Any, List, Optional

from core.grid import Grid, Net
from assistant.validator import validate_action, ValidationResult
from executor.sandbox import run_in_sandbox, SandboxResult, generate_preview_image
from executor.versioning import VersionManager
from executor import executor


class BoardController:
    """
    Single entry point for all board operations.
    Enforces: validate -> preview -> confirm -> apply -> version.
    """

    def __init__(self, grid: Grid, nets: List[Net]):
        self.grid = grid
        self.nets = nets
        self.versions = VersionManager(grid, nets)
        self._pending_sandbox: Optional[SandboxResult] = None
        self._pending_action_id: Optional[str] = None
        self._pending_payload: Optional[Dict[str, Any]] = None

    # ---------------------------------------------------------- #
    # PREVIEW (Step 1: Validate + Sandbox)
    # ---------------------------------------------------------- #

    def preview(self, action_id: str,
                payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate an action and run it in sandbox.
        Returns preview result (never modifies real board).
        """
        # 1. Validate
        validation = validate_action(action_id, payload)
        if not validation.valid:
            self._pending_sandbox = None
            return {
                "status": "validation_error",
                "errors": [
                    {"field": e.field, "msg": e.msg}
                    for e in validation.errors
                ],
            }

        # 2. Run in sandbox
        sandbox = run_in_sandbox(
            self.grid, self.nets,
            action_id, validation.cleaned
        )

        # 3. Store pending state for apply
        self._pending_sandbox = sandbox
        self._pending_action_id = action_id
        self._pending_payload = validation.cleaned

        # 4. Return preview result
        result = {
            "status": "preview_ready",
            "action": action_id,
            "action_result": sandbox.action_result,
            "drc": sandbox.drc_report,
            "drc_passed": sandbox.drc_passed,
        }

        return result

    def get_preview_image(self) -> Optional[bytes]:
        """Get the preview image from the last sandbox run."""
        if self._pending_sandbox is None:
            return None
        return generate_preview_image(
            self._pending_sandbox.preview_grid,
            self._pending_sandbox.routing_result,
        )

    def get_preview_grid(self) -> Optional[Grid]:
        """Get the preview grid from the last sandbox run."""
        if self._pending_sandbox is None:
            return None
        return self._pending_sandbox.preview_grid

    # ---------------------------------------------------------- #
    # APPLY (Step 2: Commit to real board)
    # ---------------------------------------------------------- #

    def apply(self) -> Dict[str, Any]:
        """
        Apply the last previewed action to the real board.
        Creates a version backup first.
        """
        if self._pending_sandbox is None:
            return {"status": "error", "msg": "No pending preview to apply"}

        if not self._pending_sandbox.ok:
            return {
                "status": "error",
                "msg": "Cannot apply: action failed in sandbox",
                "detail": self._pending_sandbox.action_result,
            }

        # Apply to real board: replace grid state with sandbox result
        sandbox_grid = self._pending_sandbox.preview_grid

        # Copy sandbox state to real grid
        self.grid.occupied = sandbox_grid.occupied.copy()
        self.grid.heat = sandbox_grid.heat.copy()
        self.grid.congestion = sandbox_grid.congestion.copy()
        self.grid.components = [deepcopy(c)
                                for c in sandbox_grid.components]
        self.grid.routed_paths = dict(sandbox_grid.routed_paths)

        # Commit version
        version_num = self.versions.commit(
            self.grid,
            self._pending_action_id,
            self._pending_payload,
            self._pending_sandbox.action_result,
        )

        # Clear pending
        action_id = self._pending_action_id
        result = self._pending_sandbox.action_result
        self._pending_sandbox = None
        self._pending_action_id = None
        self._pending_payload = None

        return {
            "status": "applied",
            "action": action_id,
            "version": version_num,
            "result": result,
        }

    # ---------------------------------------------------------- #
    # UNDO / REDO
    # ---------------------------------------------------------- #

    def undo(self) -> Dict[str, Any]:
        """Undo the last applied action."""
        restored = self.versions.undo()
        if restored is None:
            return {"status": "error", "msg": "Nothing to undo"}

        self._copy_grid_state(restored)
        return {
            "status": "ok",
            "msg": "Undone",
            "version": self.versions.version_count - 1,
        }

    def redo(self) -> Dict[str, Any]:
        """Redo a previously undone action."""
        restored = self.versions.redo()
        if restored is None:
            return {"status": "error", "msg": "Nothing to redo"}

        self._copy_grid_state(restored)
        return {
            "status": "ok",
            "msg": "Redone",
            "version": self.versions.version_count - 1,
        }

    # ---------------------------------------------------------- #
    # INFO / DRC
    # ---------------------------------------------------------- #

    def get_board_info(self) -> Dict[str, Any]:
        """Get current board state summary."""
        return executor.get_board_info(self.grid)

    def run_drc(self) -> Dict[str, Any]:
        """Run DRC on current board state."""
        return executor.get_drc_report(self.grid, self.nets)

    def get_history(self) -> List[Dict[str, Any]]:
        """Get version history."""
        return self.versions.get_history()

    def export_history_log(self) -> str:
        """Export version history as JSON."""
        return self.versions.export_log()

    # ---------------------------------------------------------- #
    # INTERNAL
    # ---------------------------------------------------------- #

    def _copy_grid_state(self, source: Grid) -> None:
        """Copy grid state from source into self.grid."""
        self.grid.occupied = source.occupied.copy()
        self.grid.heat = source.heat.copy()
        self.grid.congestion = source.congestion.copy()
        self.grid.components = [deepcopy(c)
                                for c in source.components]
        self.grid.routed_paths = dict(source.routed_paths)
