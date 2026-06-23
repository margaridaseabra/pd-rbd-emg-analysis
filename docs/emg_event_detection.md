# EMG event detection summary

The detector converts raw EMG into a moving RMS envelope, log-transforms it, normalizes it to a quiet NREM baseline, and detects bursts with hysteresis thresholding.

Typical defaults:

- RMS window: 0.25 s
- onset threshold: z >= 4
- offset threshold: z < 2
- microburst merge gap: 0.5 s
- episode merge gap: 10 s
- minimum duration: 0.10 s

A high onset threshold makes detection conservative, while the lower offset threshold and merging prevent biologically continuous events from being fragmented.
