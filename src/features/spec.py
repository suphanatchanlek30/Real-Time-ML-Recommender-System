"""The feature list: name, type, group, valid range. No heavy dependencies.

Single source of truth used by training (Pandera schemas, schema.json), the feature code and
serving (which checks that the saved model's feature order equals this list). It is kept free
of Pandera so the serving image does not need it; the Pandera schemas live in schema.py.
"""

import json
from dataclasses import asdict, dataclass

# Parameters of the feature definitions. They are written to schema.json too.
DECAY_TAU_DAYS = 7  # half-memory of the decayed counts: weight exp(-age / tau)
WINDOWS_DAYS = (1, 7, 28)  # item popularity windows
PRIOR_STRENGTH = 20  # smoothing: an item needs ~20 views before its own rate dominates
RECENT_N = 20  # length of the "recent items" list (the Redis `user:{id}:recent` key)


@dataclass(frozen=True)
class Feature:
    name: str
    dtype: str  # "int" or "float"
    group: str  # user | item | category | pair | candidate
    desc: str
    min: float | None = None
    max: float | None = None
    nullable: bool = False


FEATURES = [
    # --- user: history before `as_of` ---
    Feature("u_n_events", "int", "user", "all events", min=1),
    Feature("u_n_view", "int", "user", "view events", min=0),
    Feature("u_n_cart", "int", "user", "addtocart events", min=0),
    Feature("u_n_buy", "int", "user", "transaction events", min=0),
    Feature("u_n_sessions", "int", "user", "sessions (30 min gap)", min=1),
    Feature("u_cart_per_view", "float", "user", "addtocart / max(view, 1)", min=0),
    Feature("u_buy_per_event", "float", "user", "transaction / events", min=0, max=1),
    Feature("u_days_since_last", "float", "user", "days since last event", min=0),
    Feature("u_days_since_first", "float", "user", "days since first event", min=0),
    Feature("u_decay_interest", "float", "user", "time-decayed weighted events", min=0),
    Feature("u_decay_actions", "float", "user", "time-decayed addtocart + transaction", min=0),
    # --- item ---
    Feature("i_w_all", "int", "item", "weighted events, whole history", min=1),
    Feature("i_w_1d", "int", "item", "weighted events, last 1 day", min=0),
    Feature("i_w_7d", "int", "item", "weighted events, last 7 days", min=0),
    Feature("i_w_28d", "int", "item", "weighted events, last 28 days", min=0),
    Feature("i_trend", "float", "item", "last-day rate vs 7-day daily rate (>1 = rising)", min=0),
    Feature("i_n_view", "int", "item", "view events", min=0),
    Feature("i_n_cart", "int", "item", "addtocart events", min=0),
    Feature("i_n_buy", "int", "item", "transaction events", min=0),
    Feature("i_n_users", "int", "item", "distinct users", min=1),
    Feature("i_days_since_last", "float", "item", "days since last event", min=0),
    Feature("i_age_days", "float", "item", "days since first event", min=0),
    Feature("i_cart_rate_s", "float", "item", "smoothed addtocart per view", min=0),
    Feature("i_buy_rate_s", "float", "item", "smoothed transaction per view", min=0),
    Feature("i_categoryid", "int", "item", "category at as_of (-1 unknown)", min=-1),
    Feature("i_parent_categoryid", "int", "item", "parent category (-1 none)", min=-1),
    Feature("i_available", "int", "item", "availability at as_of (-1 unknown)", min=-1, max=1),
    # --- category (of the item) ---
    Feature("c_w_all", "int", "category", "weighted events of the category", min=1),
    Feature("c_w_7d", "int", "category", "weighted events, last 7 days", min=0),
    Feature("c_n_items", "int", "category", "items in the category", min=1),
    # --- user x item pair, computed when candidates are scored ---
    Feature("p_n_view", "int", "pair", "times this user viewed this item", min=0),
    Feature("p_n_cart", "int", "pair", "times this user carted this item", min=0),
    Feature("p_n_buy", "int", "pair", "times this user bought this item", min=0),
    Feature(
        "p_days_since_touch",
        "float",
        "pair",
        "days since the user last touched it",
        min=0,
        nullable=True,
    ),
    Feature(
        "p_cat_share",
        "float",
        "pair",
        "share of the user's events in the item's category",
        min=0,
        max=1,
    ),
    # --- candidate generation (how the item was retrieved) ---
    Feature(
        "a_score", "float", "candidate", "ALS score (null for popular fallback)", nullable=True
    ),
    Feature("a_rank", "int", "candidate", "position in the retrieved list (1 = best)", min=1),
    Feature(
        "a_source", "int", "candidate", "1 = ALS candidate, 0 = popular fallback", min=0, max=1
    ),
]
GROUPS = ("user", "item", "category", "pair", "candidate")
FEATURE_COLUMNS = [f.name for f in FEATURES]  # model input order
BY_GROUP = {g: [f for f in FEATURES if f.group == g] for g in GROUPS}


def schema_dict() -> dict:
    return {
        "version": 1,
        "params": {
            "decay_tau_days": DECAY_TAU_DAYS,
            "windows_days": list(WINDOWS_DAYS),
            "prior_strength": PRIOR_STRENGTH,
            "recent_n": RECENT_N,
        },
        "feature_order": FEATURE_COLUMNS,
        "features": [asdict(f) for f in FEATURES],
    }


def write_schema(path) -> None:
    path.write_text(json.dumps(schema_dict(), indent=2, ensure_ascii=False))
