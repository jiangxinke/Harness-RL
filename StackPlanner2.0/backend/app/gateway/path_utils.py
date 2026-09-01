"""Shared path resolution for thread virtual paths (e.g. mnt/user-data/outputs/...)."""

from pathlib import Path

from fastapi import HTTPException

from deerflow.config.paths import get_paths
from deerflow.runtime.user_context import get_effective_user_id


def resolve_thread_virtual_path(thread_id: str, virtual_path: str) -> Path:
    """Resolve a virtual path to the actual filesystem path under thread user-data.

    Args:
        thread_id: The thread ID.
        virtual_path: The virtual path as seen inside the sandbox
                      (e.g., /mnt/user-data/outputs/file.txt).

    Returns:
        The resolved filesystem path.

    Raises:
        HTTPException: If the path is invalid or outside allowed directories.
    """
    paths = get_paths()
    user_id = get_effective_user_id()
    try:
        scoped_path = paths.resolve_virtual_path(thread_id, virtual_path, user_id=user_id)
        if user_id is not None and not scoped_path.exists():
            # Keep read access to artifacts created before user-scoped thread
            # storage was enabled. Ownership is checked by the route decorator
            # before this resolver runs; this fallback only bridges the legacy
            # on-disk layout and never changes the validated path root.
            legacy_path = paths.resolve_virtual_path(thread_id, virtual_path)
            if legacy_path.exists():
                return legacy_path
        return scoped_path
    except ValueError as e:
        status = 403 if "traversal" in str(e) else 400
        raise HTTPException(status_code=status, detail=str(e))
