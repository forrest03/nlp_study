# Whisper model selection

- `tiny`: fastest and least accurate; useful for a quick pipeline test.
- `base`: small local test model.
- `small`: default balance for this project.
- `medium`: slower and more accurate, with higher memory and download cost.
- `large-v3`: largest supported option in this project; use only when the machine
  has enough memory and long processing time is acceptable.

All models run on CPU with `int8` in the current implementation.

