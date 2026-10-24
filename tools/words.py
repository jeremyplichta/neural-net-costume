"""The vocabulary the costume recognises. Order = output-neuron order."""

# (label, emoji). "other" is the catch-all class: random speech, noise, silence.
WORDS = [
    ("other", "💬"),
    ("ghost", "👻"),
    ("boo", "🎃"),
    ("spooky", "💀"),
    ("witch", "🧙"),
    ("candy", "🍬"),
    ("cat", "🐈‍⬛"),
    ("bat", "🦇"),
    ("spider", "🕷️"),
    ("skull", "☠️"),
]
LABELS = [w for w, _ in WORDS]
EMOJI = dict(WORDS)
