# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Archive and restore complete task roots, including files omitted by Git diffs."""

import hashlib
import json
import os
import shutil
import stat
import sys
import tarfile
from pathlib import Path, PurePosixPath


CHUNK_BYTES = 4 * 1024 * 1024
FORBIDDEN_ROOTS = {"/", "/tmp", "/opt", "/home", "/root", "/usr", "/etc", "/var", "/proc", "/sys", "/dev"}


def task_roots(paths: list[str]) -> list[str]:
    """Choose only initial task roots, never broad system or harness directories."""
    roots = []
    for value in sorted(set(paths), key=lambda value: (len(PurePosixPath(value).parts), value)):
        path = PurePosixPath(value)
        if not path.is_absolute() or ".." in path.parts or str(path) in FORBIDDEN_ROOTS:
            raise ValueError(f"Unsafe task snapshot root: {value}")
        if str(path).startswith(("/opt/gym-runtime", "/tmp/nemo-gym-")):
            raise ValueError(f"Harness runtime cannot be a task snapshot root: {value}")
        if not any(path == PurePosixPath(root) or PurePosixPath(root) in path.parents for root in roots):
            roots.append(str(path))
    return roots


def digest(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def freeze(roots: list[str], directory: Path) -> dict:
    """Record every regular file, directory, hard link and symlink under the task roots."""
    roots = task_roots(roots)
    directory.mkdir(parents=True, exist_ok=True)
    archive_path = directory / "state.tar.gz"
    with tarfile.open(archive_path, "w:gz", dereference=False) as archive:
        for root in roots:
            path = Path(root)
            if path.is_symlink():
                raise ValueError(f"Task root became a symlink: {root}")
            if path.exists():
                archive.add(path, arcname=root.lstrip("/"), recursive=True)
    chunks = []
    with archive_path.open("rb") as source:
        while block := source.read(CHUNK_BYTES):
            name = f"chunk-{len(chunks):06d}"
            (directory / name).write_bytes(block)
            chunks.append({"name": name, "bytes": len(block), "sha256": hashlib.sha256(block).hexdigest()})
    manifest = {"roots": roots, "archive_sha256": digest(archive_path), "chunks": chunks}
    (directory / "manifest.json").write_text(json.dumps(manifest))
    archive_path.unlink()
    return manifest


def clear_directory(path: Path) -> None:
    """Remove baseline contents, allowing changes to directories owned by this user."""
    attributes = path.stat()
    if attributes.st_uid == os.geteuid():
        path.chmod(stat.S_IMODE(attributes.st_mode) | stat.S_IRWXU)
    for child in path.iterdir():
        if child.is_symlink() or not child.is_dir():
            child.unlink()
        else:
            clear_directory(child)
            child.rmdir()


def restore(directory: Path) -> dict:
    """Replace task contents in a fresh image, retaining existing root directories."""
    manifest = json.loads((directory / "manifest.json").read_text())
    roots = task_roots(manifest["roots"])
    archive_path = directory / "state.tar.gz"
    with archive_path.open("wb") as output:
        for chunk in manifest["chunks"]:
            name = chunk["name"]
            if Path(name).name != name or not name.startswith("chunk-"):
                raise ValueError("Invalid submission chunk")
            path = directory / name
            if path.stat().st_size != chunk["bytes"] or digest(path) != chunk["sha256"]:
                raise ValueError("Submission chunk checksum mismatch")
            with path.open("rb") as source:
                shutil.copyfileobj(source, output)
    if digest(archive_path) != manifest["archive_sha256"]:
        raise ValueError("Submission archive checksum mismatch")
    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        for item in members:
            name = PurePosixPath(item.name)
            target = PurePosixPath("/") / name
            if (
                name.is_absolute()
                or ".." in name.parts
                or not any(target == PurePosixPath(root) or PurePosixPath(root) in target.parents for root in roots)
            ):
                raise ValueError("Submission member is outside the task roots")
            if not (item.isfile() or item.isdir() or item.issym() or item.islnk()):
                raise ValueError("Submission contains an unsupported filesystem entry")
        archived_directories = {"/" + item.name.rstrip("/") for item in members if item.isdir()}
        for root in roots:
            path = Path(root)
            if path.is_symlink() or path.is_file():
                path.unlink()
            elif path.exists():
                # A task user can own /workspace without being allowed to unlink
                # it from /. Keep that directory and restore its archived attrs.
                clear_directory(path)
                if root not in archived_directories:
                    path.rmdir()

        # Absolute task symlinks are preserved, but never followed while extracting
        # another member. This allows normal venv links without writing through them.
        def checked_member(item: tarfile.TarInfo, destination: str) -> tarfile.TarInfo:
            target = Path(destination) / item.name
            if any(parent.is_symlink() for parent in target.parents):
                raise ValueError("Submission member would write through a symlink")
            if item.islnk():
                link = PurePosixPath(item.linkname)
                if (
                    link.is_absolute()
                    or ".." in link.parts
                    or not any((PurePosixPath("/") / link).is_relative_to(root) for root in roots)
                ):
                    raise ValueError("Submission hardlink is outside task roots")
            return item

        # extractall delays directory ownership, modes and mtimes until children
        # exist, then restores deepest directories first (including read-only ones).
        archive.extractall(path="/", members=members, filter=checked_member)
    archive_path.unlink()
    return {"archive_sha256": manifest["archive_sha256"], "restored_roots": roots, "members": len(members)}


if __name__ == "__main__":
    action, directory, *args = sys.argv[1:]
    try:
        result = freeze(json.loads(args[0]), Path(directory)) if action == "freeze" else restore(Path(directory))
    except PermissionError as exc:
        raise RuntimeError(
            f"Submission {action} denied for uid={os.geteuid()} gid={os.getegid()} at {exc.filename!r}: "
            f"{exc.strerror}. Task roots must be readable for freeze and writable for restore by the image user."
        ) from exc
    print(json.dumps(result))
