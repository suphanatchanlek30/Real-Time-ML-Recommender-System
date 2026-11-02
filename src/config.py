"""Constants shared by features, evaluation and (later) training and serving."""

# Interest weight per event type; also the starting point for ALS confidence (tuned later)
EVENT_WEIGHTS = {"view": 1, "addtocart": 3, "transaction": 5}

# Two events of the same user further apart than this start a new session
SESSION_GAP_MINUTES = 30
