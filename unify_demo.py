#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["httpx>=0.28"]
# ///
"""unify's demo: one round of continual learning on a small model, end to end, on unify's infrastructure.

    uv run unify_demo.py --key unify_sk_...

It takes a student model, a teacher model and a standard agent benchmark, tau2-bench's retail domain, and shows
the loop making the student better. tau2-bench's stock agent stands in for your harness, changed only as you
would change yours: its model calls go to unify's gateway with your unify key. Your machine runs the benchmark;
unify serves the student, trains it and serves the result.

1. The untrained student plays 80 retail tasks twice, its calls captured by the gateway as your agent's.
2. A round starts: the teacher labels every one of the student's calls with its reasoning, the student is trained
   on the labels and published to its server, and the round checks it against the model it started from.
3. While the round trains, the untrained student plays the 34 held-out tasks twice. Their calls are marked for
   evaluation, and no round trains on them.
4. Once the trained student is served, it plays the held-out tasks twice.
5. pass^1 before and after, with a 95% interval on the gain.

Every step plays its conversations 64 at a time, and a run takes a little over an hour. Each step's results are
kept in ~/.cache/unify-demo/<agent>/, so a run that stops starts again where it stopped. Your unify key is the only
key it needs: the benchmark's simulated customer and its judge are models unify pays for, reached through the
gateway within your agent's allowance, and unify pays for the student, the teacher and the training. It needs git
and uv.
"""

import argparse
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import httpx

GATEWAY = "https://gateway.unify.ai"
API = "https://api.unify.ai"
TAU2_REPO = "https://github.com/sierra-research/tau2-bench"
TAU2_COMMIT = "fc0055dc4e0a316c3f83133267fbd6faaa770992"  # v1.0.1
CACHE = Path.home() / ".cache" / "unify-demo"
SEED = 300
# conversations played at once; the student's server takes 128, so two runs can overlap
CONCURRENCY = 64

# tau2-bench's retail tasks split once with a fixed seed: 70% the student plays before its round, 30% held out.
HISTORY = (
    "0 1 10 100 101 102 103 105 106 107 108 109 11 110 111 112 113 12 14 15 17 18 19 2 20 21 22 24 25 26 28 29 3 30 "
    "33 35 36 38 39 4 40 41 43 44 45 49 5 50 52 53 56 58 6 60 61 62 65 66 67 68 70 71 73 76 77 78 79 8 81 83 84 86 "
    "89 90 91 92 93 95 97 99"
).split()
HELDOUT = (
    "104 13 16 23 27 31 32 34 37 42 46 47 48 51 54 55 57 59 63 64 69 7 72 74 75 80 82 85 87 88 9 94 96 98"
).split()

# The student as the benchmark's agent: thinking on, and the token ids it read and wrote returned, so a round
# trains on exactly what it saw.
STUDENT_ARGS = {
    "temperature": 1.0,
    "top_p": 0.95,
    "max_tokens": 8192,
    "api_key": "unify-hosted",
    "extra_body": {
        "chat_template_kwargs": {"enable_thinking": True},
        "return_token_ids": True,
    },
}
# The simulated customer: DeepSeek-V4-Pro, open weights, thinking off, one of the models unify pays for.
USER_LLM = "openai/deepseek/deepseek-v4-pro"
USER_ARGS = {
    "temperature": 0.0,
    "max_tokens": 4096,
    "extra_body": {"reasoning": {"enabled": False}, "usage": {"include": True}},
}

# tau2-bench's own `tau2 run`, its agent's calls sent through the gateway with the unify key, the conversation's
# simulation id as the trajectory and a role. The simulated customer and the judge of retail's written assertions
# (GPT-4.1) are the gateway's provider unify, the models unify pays for, reached with the same key and not captured.
RUNNER = """
import os
import sys

import httpx
import litellm
import tau2.evaluator.evaluator_nl_assertions as nla
import tau2.runner.batch as batch
import tau2.runner.build as build
import tau2.utils.llm_utils  # noqa: F401

# tau2's llm_utils gives litellm one HTTP client of ten connections for all its conversations, which holds the model
# calls in flight to ten however many conversations play at once; tau2's own concurrency bounds them instead
LIMITS = httpx.Limits(max_connections=None, max_keepalive_connections=None)
litellm.client_session = httpx.Client(limits=LIMITS)
litellm.aclient_session = httpx.AsyncClient(limits=LIMITS)

SPONSORED = {
    "api_base": os.environ["UNIFY_GATEWAY"] + "/unify/v1",
    "api_key": "unify-sponsored",
    "extra_headers": {"X-Unify-Key": os.environ["UNIFY_API_KEY"]},
}
nla.DEFAULT_LLM_NL_ASSERTIONS = "openai/openai/gpt-4.1"
nla.DEFAULT_LLM_NL_ASSERTIONS_ARGS = {"temperature": 0.0, **SPONSORED}
_build = build.build_text_orchestrator


def build_text_orchestrator(*args, **kwargs):
    orchestrator = _build(*args, **kwargs)
    agent = orchestrator.agent
    agent.llm_args = {
        **agent.llm_args,
        "api_base": os.environ["UNIFY_GATEWAY"] + "/" + os.environ["UNIFY_PROVIDER"] + "/v1",
        "extra_headers": {
            "X-Unify-Key": os.environ["UNIFY_API_KEY"],
            "X-Unify-Trajectory": orchestrator.simulation_id,
            "X-Unify-Role": os.environ["UNIFY_ROLE"],
        },
    }
    orchestrator.user.llm_args = {**orchestrator.user.llm_args, **SPONSORED}
    return orchestrator


build.build_text_orchestrator = build_text_orchestrator
_run_tasks = batch.run_tasks


def run_tasks(*args, **kwargs):
    # each conversation in a file of its own: a text run's single results file is rewritten whole after every
    # conversation, and the student's token ids grow it to hundreds of megabytes, which stalls every other one
    return _run_tasks(*args, **{**kwargs, "results_format": "dir"})


batch.run_tasks = run_tasks
sys.argv = ["tau2"] + sys.argv[1:]
from tau2.cli import main  # noqa: E402

main()
"""


def say(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def tau2() -> Path:
    """tau2-bench at its pinned commit, installed once in the cache."""
    source = CACHE / "tau2-bench"
    if not (source / ".venv").exists():
        say("Installing tau2-bench v1.0.1 (once, a few minutes)")
        shutil.rmtree(source, ignore_errors=True)
        subprocess.run(["git", "clone", "--quiet", TAU2_REPO, str(source)], check=True)
        subprocess.run(
            ["git", "-C", str(source), "checkout", "--quiet", TAU2_COMMIT], check=True
        )
        subprocess.run(
            ["uv", "sync", "--quiet", "--python", "3.12"], cwd=source, check=True
        )
    (CACHE / "tau2_run.py").write_text(RUNNER)
    return source


def play(
    work: Path,
    name: str,
    tasks: list[str],
    model: str,
    provider: str,
    role: str,
    key: str,
    trials: int,
) -> list[dict]:
    """The student as tau2-bench's agent over ``tasks``, ``trials`` times each: each conversation's id, task,
    reward and how it ended (tau2's index of its results), kept in ``work``."""
    kept = work / f"{name}.json"
    if kept.exists():
        say(f"{name}: kept from an earlier run")
        return json.loads(kept.read_text())
    source = tau2()
    save = f"unify_demo_{work.name}_{name}"
    shutil.rmtree(source / "data" / "simulations" / save, ignore_errors=True)
    log = work / f"{name}.log"
    say(f"{name}: {len(tasks)} tasks, {trials} trials each, as {model} (log: {log})")
    with open(log, "w") as out:
        process = subprocess.Popen(
            [
                str(source / ".venv" / "bin" / "python"),
                "-I",
                str(CACHE / "tau2_run.py"),
                "run",
                "--domain",
                "retail",
                "--agent",
                "llm_agent",
                "--agent-llm",
                f"openai/{model}",
                "--agent-llm-args",
                json.dumps(STUDENT_ARGS),
                "--user",
                "user_simulator",
                "--user-llm",
                USER_LLM,
                "--user-llm-args",
                json.dumps(USER_ARGS),
                "--task-ids",
                *tasks,
                "--num-trials",
                str(trials),
                "--max-concurrency",
                str(CONCURRENCY),
                "--seed",
                str(SEED),
                "--save-to",
                save,
            ],
            cwd=source,
            stdout=out,
            stderr=subprocess.STDOUT,
            env={
                **os.environ,
                "UNIFY_GATEWAY": GATEWAY,
                "UNIFY_PROVIDER": provider,
                "UNIFY_API_KEY": key,
                "UNIFY_ROLE": role,
            },
        )
        shown = ""
        while process.poll() is None:
            time.sleep(30)
            done = re.findall(
                r"(\d+/\d+ complete\. Avg reward: \S+)", log.read_text(errors="replace")
            )
            if done and done[-1] != shown:
                shown = done[-1]
                say(f"{name}: {shown}")
    if process.returncode != 0:
        sys.exit(f"{name} failed: see {log}")
    conversations = json.loads(
        (source / "data" / "simulations" / save / "results.json").read_text()
    )["simulation_index"]
    kept.write_text(json.dumps(conversations))
    return conversations


def report_outcomes(conversations: list[dict], key: str) -> None:
    """Each conversation's reward as its trajectory's outcome, as a harness reports its own."""
    with httpx.Client(
        base_url=GATEWAY, headers={"X-Unify-Key": key}, timeout=30
    ) as client:
        for conversation in conversations:
            reward = float(conversation["reward"] or 0.0)
            client.post(
                f"/v1/trajectories/{conversation['id']}/outcome",
                json={
                    "reward": reward,
                    "success": reward == 1.0,
                    "task_id": conversation["task_id"],
                    "termination_reason": conversation["termination_reason"],
                },
            ).raise_for_status()
            client.post(
                f"/v1/trajectories/{conversation['id']}/complete",
                json={"reason": conversation["termination_reason"]},
            ).raise_for_status()


def pass1(conversations: list[dict]) -> dict[str, float]:
    """Each task's mean reward over its trials."""
    rewards: dict[str, list[float]] = {}
    for conversation in conversations:
        rewards.setdefault(str(conversation["task_id"]), []).append(
            float(conversation["reward"] or 0.0)
        )
    return {task: sum(r) / len(r) for task, r in rewards.items()}


def gain(
    before: dict[str, float], after: dict[str, float]
) -> tuple[float, float, float]:
    """The mean gain in pass^1 points, paired by task, with its 95% bootstrap interval over tasks."""
    diffs = [100 * (after[t] - before[t]) for t in sorted(set(before) & set(after))]
    rng = random.Random(20261014)
    draws = sorted(
        sum(rng.choice(diffs) for _ in diffs) / len(diffs) for _ in range(10_000)
    )
    return sum(diffs) / len(diffs), draws[250], draws[9749]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--key",
        default=os.environ.get("UNIFY_API_KEY"),
        help="your unify API key (unify_sk_...)",
    )
    parser.add_argument(
        "--agent", help="the agent's slug, when the key reaches several"
    )
    parser.add_argument(
        "--student",
        default="Qwen/Qwen3.5-0.8B",
        help="Qwen/Qwen3.5-0.8B or Qwen/Qwen3.5-2B",
    )
    parser.add_argument(
        "--teacher", default="qwen/qwen3.5-397b-a17b", help="an OpenRouter model id"
    )
    parser.add_argument("--trials", type=int, default=2, help="plays of each task")
    a = parser.parse_args()
    if not a.key:
        sys.exit("Give your unify API key with --key or UNIFY_API_KEY.")

    api = httpx.Client(
        base_url=API, headers={"Authorization": f"Bearer {a.key}"}, timeout=60
    )
    agents = api.get("/v1/agents")
    agents.raise_for_status()
    slugs = [agent["slug"] for agent in agents.json()["data"]]
    slug = a.agent or (slugs[0] if len(slugs) == 1 else None)
    if slug not in slugs:
        sys.exit(
            f"Name the agent with --agent: this key reaches {', '.join(slugs) or 'no agent'}."
        )
    work = CACHE / slug
    work.mkdir(parents=True, exist_ok=True)
    provider = a.student.rsplit("/", 1)[-1].lower()
    base = f"{provider}-base"
    say(f"Agent {slug}: student {a.student}, teacher {a.teacher}")

    history = play(work, "history", HISTORY, base, provider, "actor", a.key, a.trials)
    if not (work / "history.reported").exists():
        report_outcomes(history, a.key)
        (work / "history.reported").touch()

    started = work / "round.json"
    if not started.exists():
        response = api.post(
            f"/v1/agents/{slug}/rounds",
            json={"student": a.student, "teacher": a.teacher},
        )
        response.raise_for_status()
        started.write_text(json.dumps(response.json()))
    name = json.loads(started.read_text())["name"]
    say(
        f"Round {name}: the teacher labels the student's calls, then the student trains;"
        " meanwhile the untrained student plays the held-out tasks"
    )
    # the baseline needs only the untrained student, so it plays while the round runs; the round trains on
    # the history alone (role actor), never on these calls (role eval)
    before = play(work, "before", HELDOUT, base, provider, "eval", a.key, a.trials)
    seen: set[str] = set()
    while True:
        response = api.get(f"/v1/agents/{slug}/rounds/{name}")
        response.raise_for_status()
        round_ = response.json()
        for stage, numbers in round_["stages"].items():
            if numbers is not None and stage not in seen:
                seen.add(stage)
                say(f"Round {name}: {stage} done")
        if round_["state"] == "failed":
            sys.exit(f"Round {name} failed: {round_['error']}")
        # the trained student is published before the round's own check of it ends, which this measurement
        # does not need
        if round_["model"]:
            break
        time.sleep(30)
    model = round_["model"]

    say(f"Waiting for the student's server to serve {model}")
    with httpx.Client(
        base_url=GATEWAY,
        headers={"X-Unify-Key": a.key},
        timeout=180,
        follow_redirects=True,
    ) as gw:
        while True:
            # the server may be starting, which takes a few minutes after it has been idle, and a request to it
            # can outlast the timeout meanwhile
            try:
                listed = gw.get(f"/{provider}/v1/models")
            except httpx.TimeoutException:
                continue
            if listed.status_code == 200 and model in [
                m["id"] for m in listed.json()["data"]
            ]:
                break
            time.sleep(20)
    after = play(work, "after", HELDOUT, model, provider, "eval", a.key, a.trials)

    b, t = pass1(before), pass1(after)
    mean, low, high = gain(b, t)
    print()
    print(
        f"tau2-bench retail, {len(HELDOUT)} held-out tasks, {a.trials} trials each, pass^1:"
    )
    print(f"  {a.student} before its round  {100 * sum(b.values()) / len(b):5.1f}")
    print(
        f"  after one round taught by {a.teacher}  {100 * sum(t.values()) / len(t):5.1f}"
    )
    print(f"  gain  {mean:+.1f} points (95% interval {low:+.1f} to {high:+.1f})")


if __name__ == "__main__":
    main()
