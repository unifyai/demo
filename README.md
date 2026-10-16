# unify demo

One round of continual learning on a small model, start to finish, run from your laptop on unify's infrastructure.
At the end it prints how much better the model got.

unify's API is still being built, and only people we've invited can use it. You need a unify key (`unify_sk_...`)
from us, plus git and [uv](https://docs.astral.sh/uv/). You don't need any other account or a GPU, because we serve
and train every model in the demo and pay for all of them.

```bash
git clone https://github.com/unifyai/demo && cd demo
UNIFY_API_KEY=unify_sk_... uv run unify_demo.py
```

## What it does

[tau2-bench](https://github.com/sierra-research/tau2-bench)'s retail tasks stand in for your agent. Its stock agent
deals with a simulated customer who wants a return or an exchange, using a store's tools and policy. A task counts
as solved when the store's database ends up right and a judge model agrees the conversation said what it had to.
We changed the agent only the way you'd change your own harness to use us: its base URL points at our gateway, and
it sends your key in a header.

The student is Qwen3.5-0.8B, small enough to start well below its teacher, Qwen3.5-397B-A17B. The script:

1. has the untrained student play 80 of the tasks twice through the gateway, which records every call it makes as
   your agent's history;
2. starts a training round through our API. The teacher goes over every call the student made and writes what it
   would have thought and done at that point. The student is fine-tuned on that and put on its server, and the
   round then checks it against the model it started from;
3. has the untrained student play the other 34 tasks twice while the round trains, for a baseline (no round ever
   trains on these);
4. has the trained student play the same 34 tasks twice once it's served;
5. prints pass^1 before and after, with a 95% interval on the gain.

It takes a little over an hour, plus a few minutes the first time to install tau2-bench. The benchmark runs on your
machine and the models all run on ours. The simulated customer and the benchmark's judge are models we pay for, and
your key can use them up to an allowance that covers a couple of runs. Everything is kept in `~/.cache/unify-demo/`,
so if a run stops, start it again and it picks up where it left off.

## What we got

We ran it the way you would on 15 October 2026, with a new key on a laptop:

```text
tau2-bench retail, 34 held-out tasks, 2 trials each, pass^1:
  Qwen/Qwen3.5-0.8B before its round   35.3
  after one round taught by qwen/qwen3.5-397b-a17b   61.8
  gain  +26.5 points (95% interval +11.8 to +41.2)
```

That run took an hour and 11 minutes. The round was 41 of them, and the baseline played inside it. The trained
student took another 14, three of which went on waiting for its server to start. Two earlier runs that day gained
27.9 and 26.5 points. Yours will land near there but not exactly on it. The student samples its replies, so no two
runs play out the same, and with 34 tasks the interval is wide.

That's one round on a toy setup: a small model on a public benchmark. The plumbing is the real thing, though. The
calls go through the same gateway a design partner's agent would use, and the round starts through the same API.

## Options

- `--student Qwen/Qwen3.5-2B` trains the larger student. It starts closer to its teacher, so it has less to gain.
- `--teacher <OpenRouter model id>` picks another teacher. It has to think out loud, since the student learns from
  its reasoning as well as its actions.
- `--trials N` plays each task N times rather than twice. More trials narrow the interval but take longer.
- `--agent <slug>` names the agent when your key reaches more than one.
