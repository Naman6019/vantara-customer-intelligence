"""Synthesise the walkthrough's background music: a calm, royalty-free ambient bed.

    python music/make_track.py [seconds]      (from video/, with numpy + scipy)

Writes public/music/background.mp3 (needs ffmpeg). Everything is generated here, so
there is no licence to track. The track is a slow I-vi-IV-V in D major: a warm,
filtered pad, a soft sine bass, and a sparse plucked pattern from the second cycle on,
all through a short reverb. It is mixed under the voice in src/Walkthrough.tsx.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, oaconvolve, sosfilt

SR = 44_100
BAR = 3.2  # seconds: 4 beats at 75 BPM
CHORD = 2 * BAR
OUT = Path(__file__).resolve().parent.parent / "public" / "music"

# Pad voicings (MIDI) and bass roots for D - Bm - G - A.
PROGRESSION = [
    ([50, 57, 61, 66], 38),  # Dmaj7: D3 A3 C#4 F#4, bass D2
    ([47, 54, 57, 62], 35),  # Bm7:   B2 F#3 A3 D4,  bass B1
    ([43, 55, 59, 66], 31),  # Gmaj7: G2 G3 B3 F#4,  bass G1
    ([45, 52, 61, 64], 33),  # A:     A2 E3 C#4 E4,  bass A1
]
ARP = [0, 1, 2, 3, 2, 1, 2, 3]  # chord-tone order for the plucks, one per beat


def hz(midi: float) -> float:
    return 440.0 * 2 ** ((midi - 69) / 12)


def envelope(n: int, attack: float, release: float) -> np.ndarray:
    env = np.ones(n)
    a, r = int(attack * SR), int(release * SR)
    env[:a] = np.linspace(0, 1, a) ** 2
    env[-r:] *= np.linspace(1, 0, r) ** 2
    return env


def pad_note(midi: int, seconds: float, cents: float) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    f = hz(midi) * 2 ** (cents / 1200)
    tone = np.sin(2 * np.pi * f * t) + 0.3 * np.sin(4 * np.pi * f * t) + 0.12 * np.sin(6 * np.pi * f * t)
    tremolo = 1 + 0.06 * np.sin(2 * np.pi * 0.17 * t + midi)  # slow movement
    return tone * tremolo


def pluck(midi: int, seconds: float = 1.6) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    f = hz(midi)
    tone = np.sin(2 * np.pi * f * t) + 0.25 * np.sin(4 * np.pi * f * t)
    env = np.exp(-t / 0.35) * np.minimum(1, t / 0.005)
    return tone * env


def add(track: np.ndarray, sound: np.ndarray, start: float, gain: float) -> None:
    i = int(start * SR)
    j = min(len(track), i + len(sound))
    if i < len(track):
        track[i:j] += gain * sound[: j - i]


def render(seconds: float) -> np.ndarray:
    n = int(seconds * SR)
    pad = np.zeros((n, 2))
    bass = np.zeros(n)
    plucks = np.zeros((n, 2))
    release = 2.5

    step = 0
    for k in range(int(np.ceil(seconds / CHORD))):
        notes, root = PROGRESSION[k % len(PROGRESSION)]
        start = k * CHORD
        length = CHORD + release
        env = envelope(int(length * SR), attack=1.8, release=release)
        for m in notes:
            for ch, cents in ((0, -4.0), (1, 4.0)):  # slight detune: width
                add(pad[:, ch], pad_note(m, length, cents) * env, start, 0.11)
        b = np.sin(2 * np.pi * hz(root) * np.arange(int(length * SR)) / SR)
        add(bass, b * envelope(len(b), attack=0.4, release=release), start, 0.22)

        if k >= len(PROGRESSION):  # plucks join from the second cycle
            beats = int(CHORD / (BAR / 4))
            for beat in range(beats):
                if beat % 4 == 3:  # leave space: three plucks per bar
                    continue
                m = notes[ARP[step % len(ARP)]] + 12
                pan = 0.35 if step % 2 else -0.35
                p = pluck(m)
                add(plucks[:, 0], p, start + beat * BAR / 4, 0.07 * (1 - pan))
                add(plucks[:, 1], p, start + beat * BAR / 4, 0.07 * (1 + pan))
                step += 1

    warm = butter(2, 1800, btype="low", fs=SR, output="sos")
    pad = sosfilt(warm, pad, axis=0)
    mix = pad + plucks + bass[:, None]

    # Short reverb: decaying, low-passed stereo noise as the impulse response.
    rng = np.random.default_rng(7)
    ir_t = np.arange(int(2.8 * SR)) / SR
    ir = rng.standard_normal((len(ir_t), 2)) * np.exp(-ir_t / 0.7)[:, None]
    ir = sosfilt(butter(1, 4000, btype="low", fs=SR, output="sos"), ir, axis=0)
    ir /= np.sqrt((ir**2).sum(axis=0))
    wet = np.stack([oaconvolve(mix[:, c], ir[:, c])[:n] for c in range(2)], axis=1)
    out = 0.8 * mix + 0.35 * wet

    out *= envelope(n, attack=4.0, release=8.0)[:, None]
    return out / np.abs(out).max() * 10 ** (-1 / 20)  # peak at -1 dBFS


def main() -> None:
    seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 300.0
    OUT.mkdir(parents=True, exist_ok=True)
    wav = OUT / "background.wav"
    wavfile.write(wav, SR, (render(seconds) * 32767).astype(np.int16))
    mp3 = OUT / "background.mp3"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav), "-c:a", "libmp3lame", "-b:a", "192k", str(mp3)],
        check=True,
    )
    wav.unlink()
    print(f"Wrote {mp3} ({seconds:.0f} s)")


if __name__ == "__main__":
    main()
