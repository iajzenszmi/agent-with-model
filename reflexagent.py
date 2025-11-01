from dataclasses import dataclass
from typing import Any, Callable, Dict, Tuple, List, Optional

# ---------- Generic rule + models ----------

@dataclass(frozen=True)
class Rule:
    """A simple condition–action rule."""
    condition: Callable[[Dict[str, Any]], bool]
    action: Any

def rule_match(state: Dict[str, Any], rules: List[Rule]) -> Optional[Rule]:
    for r in rules:
        if r.condition(state):
            return r
    return None

def update_state(
    state: Dict[str, Any],
    last_action: Any,
    percept: Any,
    transition_model: Callable[[Dict[str, Any], Any], Dict[str, Any]],
    sensor_model: Callable[[Any, Dict[str, Any]], Dict[str, Any]]
) -> Dict[str, Any]:
    """
    state_{t|t-1} = T(state_{t-1}, a_{t-1})
    state_{t}     = incorporate Z_t via sensor model
    """
    # Predict (apply dynamics from previous action)
    predicted = transition_model(state, last_action)
    # Correct (incorporate current percept)
    corrected = sensor_model(percept, predicted)
    return corrected

class ModelBasedReflexAgent:
    def __init__(
        self,
        transition_model: Callable[[Dict[str, Any], Any], Dict[str, Any]],
        sensor_model: Callable[[Any, Dict[str, Any]], Dict[str, Any]],
        rules: List[Rule],
        initial_state: Optional[Dict[str, Any]] = None,
        no_op: Any = None
    ):
        self.transition_model = transition_model
        self.sensor_model = sensor_model
        self.rules = rules
        self.state: Dict[str, Any] = initial_state or {}
        self.action = no_op
        self.no_op = no_op

    def perceive(self, percept: Any) -> Any:
        # 1) Keep track of current state via internal model
        self.state = update_state(
            self.state, self.action, percept,
            self.transition_model, self.sensor_model
        )
        # 2) Pick an action via rule matching
        r = rule_match(self.state, self.rules)
        self.action = r.action if r else self.no_op
        return self.action

# ---------- Tiny example: Vacuum World ----------

# World:
#   Locations: 'A', 'B'
#   Agent at loc in {'A','B'}
#   Dirt flags: dirty['A'], dirty['B'] (bool)
# Percepts from environment to agent at each time step:
#   percept = {'loc': 'A'|'B', 'dirty': True|False}

def vacuum_transition_model(state: Dict[str, Any], last_action: Any) -> Dict[str, Any]:
    """Deterministic dynamics: moving changes location; sucking removes dirt (handled by rules via sensor fusion)."""
    s = dict(state)  # shallow copy
    loc = s.get('loc')
    if last_action == 'LEFT':
        s['loc'] = 'A'
    elif last_action == 'RIGHT':
        s['loc'] = 'B'
    # Note: don't change dirt status here; we’ll trust sensor fusion + explicit rule effects.
    return s

def vacuum_sensor_model(percept: Dict[str, Any], predicted: Dict[str, Any]) -> Dict[str, Any]:
    """
    Incorporate the percept:
      - Set current location.
      - Update belief about dirt at the current location from the sensor.
      - Other location's dirt remains as believed.
    """
    s = dict(predicted)
    loc = percept['loc']
    s['loc'] = loc
    dirty = s.get('dirty', {'A': False, 'B': False})
    dirty = dict(dirty)
    dirty[loc] = percept['dirty']  # trust current sensor for current tile
    s['dirty'] = dirty
    return s

# Rules:
#   If current square is dirty -> SUCK
#   else if at A -> RIGHT
#   else if at B -> LEFT
vacuum_rules = [
    Rule(lambda st: st.get('dirty', {}).get(st.get('loc'), False) is True, 'SUCK'),
    Rule(lambda st: st.get('loc') == 'A', 'RIGHT'),
    Rule(lambda st: st.get('loc') == 'B', 'LEFT'),
]

# Driver demo
if __name__ == "__main__":
    # Initial belief (could be empty; the agent will fill via sensor model)
    agent = ModelBasedReflexAgent(
        transition_model=vacuum_transition_model,
        sensor_model=vacuum_sensor_model,
        rules=vacuum_rules,
        initial_state={'loc': 'A', 'dirty': {'A': True, 'B': True}},
        no_op='NO-OP'
    )

    # A short percept sequence (as if coming from the environment)
    percepts = [
        {'loc': 'A', 'dirty': True},   # at A, dirty
        {'loc': 'A', 'dirty': False},  # still at A (after SUCK), now clean
        {'loc': 'B', 'dirty': True},   # at B (after RIGHT), dirty
        {'loc': 'B', 'dirty': False},  # clean
        {'loc': 'A', 'dirty': False},  # back to A
    ]

    for t, z in enumerate(percepts, 1):
        action = agent.perceive(z)
        print(f"t={t} percept={z}  -> action={action}  state={agent.state}")
~ $