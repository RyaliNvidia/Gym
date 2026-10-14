# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Move a frozen task filesystem between isolated sandboxes in bounded chunks."""

import hashlib
import json
from pathlib import Path
from shlex import quote
from uuid import uuid4

from environment_servers.swe_together.runtime import PYTHON
from environment_servers.swe_together.submission_worker import task_roots
from nemo_gym.sandbox import AsyncSandbox
from nemo_gym.sandbox.utils import read_text


async def replicate_submission(
    candidate: AsyncSandbox, judge: AsyncSandbox, *, roots: list[str], artifact_dir: Path
) -> dict:
    """Copy complete initial task roots only after the candidate close receipt."""
    roots = task_roots(roots)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    remote = "/tmp/gym-swet-submission-" + uuid4().hex
    worker = Path(__file__).with_name("submission_worker.py")
    try:
        await candidate.upload(worker, remote + ".py")
        command = f"{quote(PYTHON)} {quote(remote + '.py')} freeze {quote(remote)} {quote(json.dumps(roots))}"
        result = await candidate.exec(command, timeout_s=600)
        if result.return_code or result.error_type:
            raise RuntimeError("Submission freeze failed: " + (result.stderr or result.stdout or ""))
        raw = await read_text(candidate, path=remote + "/manifest.json")
        manifest = json.loads(raw)
        (artifact_dir / "manifest.json").write_text(raw)
        await judge.exec("mkdir -p " + quote(remote), timeout_s=15)
        await judge.upload(worker, remote + ".py")
        await judge.upload(artifact_dir / "manifest.json", remote + "/manifest.json")
        for chunk in manifest["chunks"]:
            name = chunk["name"]
            if Path(name).name != name or not name.startswith("chunk-"):
                raise ValueError("Invalid submission chunk path")
            local = artifact_dir / name
            await candidate.download(remote + "/" + name, local)
            if (
                local.stat().st_size != chunk["bytes"]
                or hashlib.sha256(local.read_bytes()).hexdigest() != chunk["sha256"]
            ):
                raise ValueError("Downloaded submission chunk checksum mismatch")
            await judge.upload(local, remote + "/" + name)
        result = await judge.exec(f"{quote(PYTHON)} {quote(remote + '.py')} restore {quote(remote)}", timeout_s=600)
        if result.return_code or result.error_type:
            raise RuntimeError("Submission restore failed: " + (result.stderr or result.stdout or ""))
        receipt = json.loads(result.stdout.strip().splitlines()[-1])
        if receipt["archive_sha256"] != manifest["archive_sha256"]:
            raise ValueError("Restored submission checksum differs")
        (artifact_dir / "restore.json").write_text(json.dumps(receipt, indent=2))
        return receipt
    finally:
        # Sandbox teardown is the fallback if a provider is unreachable here.
        for sandbox in (candidate, judge):
            try:
                await sandbox.exec("rm -rf -- " + quote(remote) + " " + quote(remote + ".py"), timeout_s=15)
            except Exception:
                pass
