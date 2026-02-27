"""
FormaCore AI - workflow/memory.py
Project Memory Log: captures tacit knowledge by automatically logging
what changed, why, and the result. Makes design decisions traceable.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional


# ------------------------------------------------------------------ #
# Data structures
# ------------------------------------------------------------------ #

@dataclass
class MemoryEntry:
    """A single event in the project memory timeline."""
    timestamp: str
    event_type: str         # optimization, assistant, parameter, export, note
    title: str
    description: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)


# ------------------------------------------------------------------ #
# Project Memory
# ------------------------------------------------------------------ #

class ProjectMemory:
    """
    Accumulates design events during a session.
    Stored in Streamlit session_state for persistence across reruns.
    """

    def __init__(self):
        self.entries: List[MemoryEntry] = []
        self._created = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    @property
    def count(self) -> int:
        return len(self.entries)

    # ----- Logging helpers ----- #

    def log_optimization(
        self,
        naive_routed: int,
        naive_total: int,
        ga_routed: int,
        ga_total: int,
        naive_vias: int,
        ga_vias: int,
        naive_trace: int,
        ga_trace: int,
        ga_generations: int,
        ga_time: float,
        weights: Optional[Dict[str, float]] = None,
    ) -> MemoryEntry:
        via_red = (
            (1 - ga_vias / naive_vias) * 100
            if naive_vias > 0 else 0
        )
        len_red = (
            (1 - ga_trace / naive_trace) * 100
            if naive_trace > 0 else 0
        )

        entry = MemoryEntry(
            timestamp=_now(),
            event_type='optimization',
            title='Routing optimization completed',
            description=(
                f"GA ran {ga_generations} generations in {ga_time:.1f}s. "
                f"Routing: {ga_routed}/{ga_total} nets "
                f"(naive: {naive_routed}/{naive_total}). "
                f"Via reduction: {via_red:.0f}%. "
                f"Trace reduction: {len_red:.0f}%."
            ),
            metrics={
                'naive_routed': naive_routed,
                'naive_total': naive_total,
                'ga_routed': ga_routed,
                'ga_total': ga_total,
                'via_reduction_pct': round(via_red, 1),
                'trace_reduction_pct': round(len_red, 1),
                'ga_generations': ga_generations,
                'ga_time_s': round(ga_time, 1),
                'weights': weights,
            },
            tags=['optimization', 'ga'],
        )
        self.entries.append(entry)
        return entry

    def log_assistant_action(
        self,
        action_id: str,
        payload: Dict[str, Any],
        result: Dict[str, Any],
    ) -> MemoryEntry:
        status = result.get('status', 'unknown')
        version = result.get('version', '?')

        desc_parts = [f"Action: {action_id}"]
        for k, v in payload.items():
            desc_parts.append(f"{k}={v}")
        desc_parts.append(f"Result: {status}")
        if status == 'applied':
            desc_parts.append(f"(version {version})")

        entry = MemoryEntry(
            timestamp=_now(),
            event_type='assistant',
            title=f'Assistant: {action_id}',
            description='. '.join(desc_parts) + '.',
            metrics={
                'action': action_id,
                'payload': _make_serializable(payload),
                'status': status,
                'version': version,
            },
            tags=['assistant', action_id],
        )
        self.entries.append(entry)
        return entry

    def log_parameter_change(
        self,
        param_name: str,
        old_value: Any,
        new_value: Any,
    ) -> MemoryEntry:
        entry = MemoryEntry(
            timestamp=_now(),
            event_type='parameter',
            title=f'Parameter changed: {param_name}',
            description=(
                f"{param_name} changed from {old_value} to {new_value}."
            ),
            metrics={
                'parameter': param_name,
                'old': old_value,
                'new': new_value,
            },
            tags=['parameter', param_name],
        )
        self.entries.append(entry)
        return entry

    def log_export(
        self,
        export_type: str,
        filename: str,
    ) -> MemoryEntry:
        entry = MemoryEntry(
            timestamp=_now(),
            event_type='export',
            title=f'Exported: {export_type}',
            description=f"Generated {export_type} export: {filename}.",
            metrics={'type': export_type, 'filename': filename},
            tags=['export', export_type],
        )
        self.entries.append(entry)
        return entry

    def log_note(self, title: str, note: str) -> MemoryEntry:
        entry = MemoryEntry(
            timestamp=_now(),
            event_type='note',
            title=title,
            description=note,
            metrics={},
            tags=['note'],
        )
        self.entries.append(entry)
        return entry

    # ----- Retrieval ----- #

    def get_timeline(self) -> List[Dict[str, Any]]:
        """Return all entries as serializable dicts for display."""
        return [
            {
                'timestamp': e.timestamp,
                'type': e.event_type,
                'title': e.title,
                'description': e.description,
                'tags': e.tags,
            }
            for e in self.entries
        ]

    def get_summary(self) -> str:
        """Generate a human-readable session summary."""
        if not self.entries:
            return "No design events recorded yet."

        lines = [f"Session started: {self._created}"]
        lines.append(f"Total events: {len(self.entries)}")

        # Count by type
        type_counts: Dict[str, int] = {}
        for e in self.entries:
            type_counts[e.event_type] = type_counts.get(e.event_type, 0) + 1

        for etype, count in type_counts.items():
            lines.append(f"  {etype}: {count}")

        lines.append("")
        lines.append("Timeline:")
        for e in self.entries:
            lines.append(f"  [{e.timestamp}] {e.title}")
            lines.append(f"    {e.description}")

        return '\n'.join(lines)

    def export_json(self) -> str:
        """Export the full memory log as JSON."""
        data = {
            'session_started': self._created,
            'exported_at': _now(),
            'total_events': len(self.entries),
            'entries': [
                {
                    'timestamp': e.timestamp,
                    'event_type': e.event_type,
                    'title': e.title,
                    'description': e.description,
                    'metrics': _make_serializable(e.metrics),
                    'tags': e.tags,
                }
                for e in self.entries
            ],
        }
        return json.dumps(data, indent=2)


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def _now() -> str:
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def _make_serializable(obj: Any) -> Any:
    """Make a value JSON-serializable."""
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    elif isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    else:
        return str(obj)
