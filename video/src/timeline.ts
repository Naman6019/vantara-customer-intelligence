// Single source of truth for scene order and length (30 fps).
//
// Each section is sized to its voice recording (public/voice/, normalised to -16 LUFS)
// so it ends about 2 s after the narration. Reveal timings inside scenes follow the
// sentence starts in those recordings, offset by VOICE_LEAD_FRAMES. Footage scenes are
// cut where the narration ends; the clips themselves run a little longer.

export const FPS = 30;
export const WIDTH = 1920;
export const HEIGHT = 1080;
export const TRANSITION_FRAMES = 15;
/** Each voice starts this long into its section, once the crossfade has finished. */
export const VOICE_LEAD_FRAMES = 15;

export const SCENE_FRAMES = {
  Title: 6 * FPS,
  Problem: 26 * FPS,
  Pipeline: 37 * FPS,
  Results: 49.5 * FPS,
  AtRisk: 668, // all of public/footage/at_risk.mp4
  Planner: 900, // all of public/footage/planner.mp4
  Customer360: 43 * FPS, // public/footage/customer360.mp4, ends on SHAP and LIME
  ApiPredict: 17.5 * FPS, // public/footage/api_predict.mp4, cut after the response
  Terminal: 9 * FPS,
  Closing: 26 * FPS,
} as const;

export type SceneId = keyof typeof SCENE_FRAMES;

// Script sections, in order, with the scenes each one covers and their voice file in
// public/voice/ (null to preview without narration).
export const SECTIONS: { title: string; scenes: SceneId[]; voice: string | null }[] = [
  { title: "1. The problem", scenes: ["Title", "Problem"], voice: "01-problem.mp3" },
  { title: "2. Data and pipeline", scenes: ["Pipeline"], voice: "02-data-pipeline.mp3" },
  { title: "3. Results, honestly", scenes: ["Results"], voice: "03-results.mp3" },
  { title: "4. Where the money goes", scenes: ["AtRisk", "Planner"], voice: "04-money.mp3" },
  { title: "5. One customer, explained", scenes: ["Customer360"], voice: "05-customer.mp3" },
  { title: "6. Fast, and easy to run", scenes: ["ApiPredict", "Terminal"], voice: "06-speed.mp3" },
  { title: "7. What's next", scenes: ["Closing"], voice: "07-next.mp3" },
];

export const SCENE_ORDER: SceneId[] = SECTIONS.flatMap((s) => s.scenes);

// Background music: "Cruising" by Dosi & Aisake [NCS Release], looped to ~5 min with a
// crossfade (public/music/ncs/cruising_bed.mp3, see README). The track masters at about
// -11 LUFS; these gains put it near -20 LUFS on its own and about -28 LUFS under the voice.
export const MUSIC = {
  file: "music/ncs/cruising_bed.mp3",
  credit: "Music: Dosi & Aisake – Cruising [NCS Release] · ncs.io/Cruising",
  volumeAlone: 0.34,
  volumeUnderVoice: 0.14,
  fadeInSeconds: 2,
  fadeOutSeconds: 4,
};

/** Frame at which each scene starts on the main timeline (fades overlap neighbours). */
export const sceneStart = (id: SceneId): number => {
  let start = 0;
  for (const scene of SCENE_ORDER) {
    if (scene === id) return start;
    start += SCENE_FRAMES[scene] - TRANSITION_FRAMES;
  }
  throw new Error(`Unknown scene ${id}`);
};

export const TOTAL_FRAMES =
  SCENE_ORDER.reduce((sum, id) => sum + SCENE_FRAMES[id], 0) -
  TRANSITION_FRAMES * (SCENE_ORDER.length - 1);
