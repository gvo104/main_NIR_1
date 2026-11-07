"""Sequential real-endpoint checks. Intended to be invoked only with VPN enabled."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from .aide_policy import (
    SelectableParentAgent,
    ensure_aide_user_message,
    first_good_node,
)
from .config import LLMConfig, load_llm_config
from .docker_executor import DockerExecutor, ExecutorLimits
from .llm_client import create_openai_client, run_chat_smoke, run_tool_smoke


class LiveStageError(RuntimeError):
    pass


def _live_experiment_name() -> str:
    """Avoid reusing a partially prepared AIDE workspace after a failed run."""

    return f"stage0-live-{uuid4().hex[:8]}"


def _write_result(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _error_category(exc: Exception) -> str:
    name = type(exc).__name__.lower()
    if "authentication" in name or "permission" in name:
        return "authentication"
    if "connection" in name or "timeout" in name:
        return "connectivity"
    if "badrequest" in name or isinstance(exc, (json.JSONDecodeError, ValueError)):
        return "response_format"
    if "container" in name or "docker" in name:
        return "execution"
    return "integration"


def _record_error(
    result: dict[str, Any], stage: str, exc: Exception, config: LLMConfig
) -> None:
    message = str(exc)
    for sensitive, replacement in (
        (config.lite_llm_token, "***"),
        (config.server_address, "<server>"),
    ):
        if sensitive:
            message = message.replace(sensitive, replacement)
    result["checks"][stage] = {
        "status": "BLOCKED",
        "category": _error_category(exc),
        "error_type": type(exc).__name__,
        "message": message,
    }


def _run_aide(
    config: LLMConfig,
    artifact_root: Path,
    *,
    aide_max_tokens: int,
    image: str,
    timeout: float,
) -> dict[str, Any]:
    from aide.agent import add_task_metric, determine_task_metric
    from aide.backend import backend_openai
    from aide.journal import Journal
    from aide.utils.config import (
        _load_cfg,
        load_task_desc,
        prep_agent_workspace,
        prep_cfg,
        save_run,
    )

    repo_root = Path(__file__).resolve().parents[2]
    data_dir = repo_root / "third_party/aideml/aide/example_tasks/house_prices"
    logs_dir = artifact_root / "logs"
    workspaces_dir = artifact_root / "workspaces"
    logs_dir.mkdir(parents=True, exist_ok=True)
    workspaces_dir.mkdir(parents=True, exist_ok=True)

    cfg = _load_cfg(use_cli_args=False)
    cfg.data_dir = str(data_dir)
    cfg.goal = "Predict house sale prices from the supplied tabular data."
    cfg.eval = "Minimize root mean squared error using a fast validation split."
    cfg.log_dir = str(logs_dir)
    cfg.workspace_dir = str(workspaces_dir)
    cfg.exp_name = _live_experiment_name()
    cfg.preprocess_data = False
    cfg.copy_data = True
    cfg.generate_report = False
    cfg.exec.timeout = 120
    cfg.agent.steps = 2
    cfg.agent.k_fold_validation = 1
    cfg.agent.data_preview = True
    cfg.agent.metric_maximize = False
    cfg.agent.search.num_drafts = 1
    cfg.agent.search.debug_prob = 1.0
    cfg.agent.search.max_debug_depth = 1
    for stage in (cfg.agent.code, cfg.agent.feedback):
        stage.model = config.llm_model
        stage.provider = "openai"
        stage.temp = config.temperature
        stage.max_tokens = aide_max_tokens
        stage.reasoning_effort = None

    cfg = prep_cfg(cfg)
    task_desc = load_task_desc(cfg)
    task_metric = determine_task_metric(task_desc, cfg.agent)
    task_desc = add_task_metric(task_desc, task_metric)
    prep_agent_workspace(cfg)

    journal = Journal(metric_maximize=False)
    agent = SelectableParentAgent(
        task_desc=task_desc,
        cfg=cfg,
        journal=journal,
        parent_selector=first_good_node,
    )
    executor = DockerExecutor(
        cfg.workspace_dir,
        image=image,
        limits=ExecutorLimits(timeout_seconds=120),
    )

    with config.aide_environment(), ensure_aide_user_message():
        backend_openai._setup_custom_client()
        backend_openai._custom_client = create_openai_client(config, timeout=timeout)
        for _ in range(2):
            agent.step(exec_callback=executor)
            save_run(cfg, journal)

    if len(journal) != 2:
        raise ValueError(f"expected two AIDE nodes, got {len(journal)}")
    if journal[1].parent is not journal[0]:
        raise ValueError("the second AIDE node is not linked to the first")

    log_dir = Path(cfg.log_dir)
    required = ("journal.json", "tree_plot.html", "best_solution.py")
    missing = [name for name in required if not (log_dir / name).is_file()]
    if missing:
        raise ValueError(f"AIDE did not save required artifacts: {', '.join(missing)}")

    return {
        "status": "PASS",
        "nodes": len(journal),
        "parent_link": True,
        "log_dir": str(log_dir.relative_to(repo_root)),
        "node_statuses": ["buggy" if node.is_buggy else "good" for node in journal],
    }


def run_all_live(
    *,
    env_file: str | Path,
    settings_file: str | Path | None = None,
    artifact_root: str | Path = "artifacts/stage0/live",
    aide_max_tokens: int | None = None,
    image: str | None = None,
    timeout: float | None = None,
) -> dict[str, Any]:
    if aide_max_tokens is not None and aide_max_tokens <= 0:
        raise ValueError("aide_max_tokens must be positive")
    if timeout is not None and timeout <= 0:
        raise ValueError("timeout must be positive")

    root = Path(__file__).resolve().parents[2]
    artifacts = (root / artifact_root).resolve()
    result_path = artifacts / "result.json"
    config = load_llm_config(env_file, settings_file, repo_root=root)
    resolved_aide_max_tokens = aide_max_tokens or config.aide_max_tokens
    resolved_timeout = timeout or config.request_timeout
    resolved_image = image or config.executor_image
    result: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": config.llm_model,
        "server": "redacted",
        "checks": {},
    }

    client = create_openai_client(config, timeout=resolved_timeout)
    try:
        chat = run_chat_smoke(client, config)
        result["checks"]["chat_completion"] = {
            "status": "PASS",
            "response_chars": len(chat),
        }
    except Exception as exc:
        _record_error(result, "chat_completion", exc, config)
        _write_result(result_path, result)
        raise LiveStageError("chat completion check failed") from exc
    _write_result(result_path, result)

    try:
        arguments = run_tool_smoke(client, config)
        result["checks"]["tool_calling"] = {
            "status": "PASS",
            "answer_chars": len(arguments["answer"]),
        }
    except Exception as exc:
        _record_error(result, "tool_calling", exc, config)
        _write_result(result_path, result)
        raise LiveStageError("tool-calling check failed; AIDE was not started") from exc
    _write_result(result_path, result)

    try:
        result["checks"]["aide_two_steps"] = _run_aide(
            config,
            artifacts,
            aide_max_tokens=resolved_aide_max_tokens,
            image=resolved_image,
            timeout=resolved_timeout,
        )
    except Exception as exc:
        _record_error(result, "aide_two_steps", exc, config)
        _write_result(result_path, result)
        raise LiveStageError("AIDE two-step check failed") from exc

    _write_result(result_path, result)
    token = config.lite_llm_token
    for path in artifacts.rglob("*"):
        if path.is_file() and token.encode() in path.read_bytes():
            raise LiveStageError(f"secret token leaked into artifact: {path}")
    return result
