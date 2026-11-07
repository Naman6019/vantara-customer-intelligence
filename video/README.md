# Vantara walkthrough video

The Day 20 "recorded walkthrough" (PRD §17), built with [Remotion](https://www.remotion.dev/).
The finished video is [`docs/walkthrough.mp4`](../docs/walkthrough.mp4); this folder is its source:
animated scenes for the story and results, plus real screen recordings of the running
Docker stack. Narration follows [script.md](script.md).

## Layout

| Path | What it is |
|---|---|
| `script.md` | Narration, section by section, with what's on screen |
| `src/timeline.ts` | Scene order and lengths, and the voice file for each section |
| `src/scenes/` | One component per scene; footage zooms and callouts are in `FootageScenes.tsx` |
| `capture/capture.mjs` | Records the live footage into `public/footage/` (not in git) |
| `public/terminal/` | Real `docker compose` and `pytest` output shown in §6 |

## Commands

```console
npm i                                   # install
npm run dev                             # Remotion Studio: preview and tweak
npx remotion render Walkthrough out/walkthrough.mp4
```

Re-record footage (needs the stack running: `docker compose up --build -d` in the parent
folder, and Chrome installed):

```console
node capture/capture.mjs                # every scene
node capture/capture.mjs customer360    # just one
```

If a clip's length changes, update its frame count in `src/timeline.ts`.

## Background music

"Cruising" by Dosi & Aisake [NCS Release] (https://ncs.io/Cruising), from NoCopyrightSounds.
It isn't in git. To rebuild the ~5-minute bed from the downloaded track:

```console
ffmpeg -i public/music/ncs/cruising.mp3 -i public/music/ncs/cruising.mp3 -filter_complex "[0:a]atrim=0:157,asetpts=PTS-STARTPTS[a];[1:a]asetpts=PTS-STARTPTS[b];[a][b]acrossfade=d=4:c1=tri:c2=tri" -c:a libmp3lame -b:a 256k public/music/ncs/cruising_bed.mp3
```

NCS requires a credit wherever the video is published. Paste this into the description:

```
Song: Dosi & Aisake - Cruising [NCS Release]
Music provided by NoCopyrightSounds
Free Download/Stream: http://ncs.io/Cruising
Watch: http://ncs.lnk.to/CruisingAT/youtube
```

NCS's usage policy asks student and indie film projects to fill in its commercial
licensing form (free in most cases) before using its music outside YouTube or Twitch.
Volume, fades and the on-screen credit are set by `MUSIC` in `src/timeline.ts`.
`music/make_track.py` synthesises a licence-free alternative instead.

## Adding the voice

The narration is ElevenLabs TTS generated from `elevenlabs/*.txt`. The raw exports in
`elevenlabs-mp3/` and the normalised copies in `public/voice/` stay out of git.

1. Normalise each export to -16 LUFS into `public/voice/`, using the names in
   `SECTIONS` (`src/timeline.ts`), e.g.
   `ffmpeg -i elevenlabs-mp3/01-problem.mp3 -af loudnorm=I=-16:TP=-1.5:LRA=11 -c:a libmp3lame -b:a 192k public/voice/01-problem.mp3`
2. Keep each section in `SCENE_FRAMES` at least 0.5 s (the voice lead) plus its
   recording long, ideally about 2 s more.
3. If a recording changes, re-time the reveals: `ffmpeg -i <file> -af silencedetect=n=-40dB:d=0.28 -f null -`
   lists the pauses between sentences, and the scene files note which line each cue
   follows.

Remotion is free for individuals; some companies need a license, see
[the terms](https://github.com/remotion-dev/remotion/blob/main/LICENSE.md).
