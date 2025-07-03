from typing import Callable, Dict, Tuple, Union, List, Optional
import ast



SilenceSegment = Tuple[float, float]
PROBABILTY_THRESHOLD = 0.10

DecisionKey = Tuple[str, str, int]
DecisionValue = Tuple[str, Union[None, Callable[..., float]]]

decision_table: Dict[DecisionKey, DecisionValue] = {
    ("before", "before", 0): ("merge", None),
    ("before", "inside", 0): ("before_inside_0", lambda *, next_start, silences, ni, **_: (silences[ni][0] + next_start) / 2),
    ("before", "before", 1): ("before_before_1", lambda *, silences, ni, **__: (silences[ni - 1][0] + silences[ni - 1][1]) / 2),
    ("before", "inside", 1): ("far", None),

    ("inside", "before", 0): ("bad", None),
    ("inside", "inside", 0): ("merge", None),
    ("inside", "before", 1): ("inside_before_1", lambda *, current_end, silences, ni, **__: (current_end + silences[ni - 1][1]) / 2),
    ("inside", "inside", 1): ("far", None),
}

def get_silence_state(word_time: float, silences: List[SilenceSegment]) -> Tuple[str, Optional[int]]:
    
    for i, (start, end) in enumerate(silences):
        if word_time < start:
            return "before", i
        elif start <= word_time <= end:
            return "inside", i
    return "after", len(silences) - 1

def decide_state_transition(
    current_end: float,
    next_start: float,
    silences: List[SilenceSegment]
) -> Tuple[str, Optional[float]]:
    current_state, ci = get_silence_state(current_end, silences)
    next_state, ni = get_silence_state(next_start, silences)

    index_diff = ni - ci

    key = (current_state, next_state, index_diff)
    result = decision_table.get(key, ("unknown", None))

    if callable(result[1]):
        return result[0], result[1](
            current_end=current_end,
            next_start=next_start,
            silences=silences,
            ni=ni
        )
    else:
        return result
    
def read_silence_segments(file_path: str) -> List[SilenceSegment]:
    with open(file_path, 'r') as f:
        content = f.read()

    # Safely evaluate the string to a Python list of tuples
    data = ast.literal_eval(content)
    return data

if __name__ == '__main__':
    file = "output_repo\\brachot4\\silences\\Bsafa_Brura-01_BR-12.srt"
    data = read_silence_segments(f"{file}.silences") 
    result, new_time = decide_state_transition(33.510, 34.080, data)   
    print(f"{result} - {new_time}")  
    