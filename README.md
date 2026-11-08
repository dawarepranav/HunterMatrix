# HunterMatrix

Heads-up short-deck 5-card PLO bot by **Beta Dancers**.

HunterMatrix plays 36-card short deck (ranks 6 through ace), five hole cards, pot-limit betting. It serves decisions over HTTP so a match judge can deal hands to it.

## How it plays

On each decision the bot builds a range matrix: sampled opponent hands against sampled board runouts. It reweights that range from the actions this opponent has taken, learns their fold and re-raise frequencies from showdowns, and picks the action with the highest expected value in chips: fold, check/call, half-pot raise, or pot raise.

Precomputed equity tables live in `brain/data/`.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install fastapi uvicorn numpy
python agent.py
```

The server listens on port 5008. Set `PORT` to use another port:

```bash
PORT=5010 python agent.py
```

`PARTICIPANT` is `TeamHunter` and `AGENT_NAME` is `HunterMatrix` in `agent.py`.

## Layout

| Path | Role |
| --- | --- |
| `agent.py` | HTTP server and safety shell around the strategy |
| `brain/strategy.py` | Action selection |
| `brain/engine.py` | Vectorized equity estimates |
| `brain/tables.py` | Lookup tables for 5-card hands |
| `brain/data/` | Precomputed tables and tuned parameters |
| `evaluator.py` | Short-deck hand evaluator |
| `game.py` | Rules, streets, and legal actions |
