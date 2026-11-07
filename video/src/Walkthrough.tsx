import { Audio } from "@remotion/media";
import { linearTiming, TransitionSeries } from "@remotion/transitions";
import { fade } from "@remotion/transitions/fade";
import React from "react";
import {
  AbsoluteFill,
  interpolate,
  Sequence,
  staticFile,
  useVideoConfig,
} from "remotion";
import { ClosingScene } from "./scenes/ClosingScene";
import {
  ApiPredictScene,
  AtRiskScene,
  Customer360Scene,
  PlannerScene,
} from "./scenes/FootageScenes";
import { PipelineScene } from "./scenes/PipelineScene";
import { ProblemScene } from "./scenes/ProblemScene";
import { ResultsScene } from "./scenes/ResultsScene";
import { TerminalScene } from "./scenes/TerminalScene";
import { TitleScene } from "./scenes/TitleScene";
import {
  MUSIC,
  SCENE_FRAMES,
  SCENE_ORDER,
  SECTIONS,
  sceneStart,
  SceneId,
  TRANSITION_FRAMES,
  VOICE_LEAD_FRAMES,
} from "./timeline";

export const SCENES: Record<SceneId, React.FC> = {
  Title: TitleScene,
  Problem: ProblemScene,
  Pipeline: PipelineScene,
  Results: ResultsScene,
  AtRisk: AtRiskScene,
  Planner: PlannerScene,
  Customer360: Customer360Scene,
  ApiPredict: ApiPredictScene,
  Terminal: TerminalScene,
  Closing: ClosingScene,
};

const hasVoice = SECTIONS.some((s) => s.voice);

/** The full walkthrough: every scene in order with short crossfades, music and voice. */
export const Walkthrough: React.FC = () => {
  const { fps, durationInFrames } = useVideoConfig();
  const level = hasVoice ? MUSIC.volumeUnderVoice : MUSIC.volumeAlone;

  return (
    <AbsoluteFill style={{ backgroundColor: "#0B1220" }}>
      <Audio
        name="Music"
        src={staticFile(MUSIC.file)}
        volume={(f) =>
          interpolate(
            f,
            [
              0,
              MUSIC.fadeInSeconds * fps,
              durationInFrames - MUSIC.fadeOutSeconds * fps,
              durationInFrames - 1,
            ],
            [0, level, level, 0],
            { extrapolateLeft: "clamp", extrapolateRight: "clamp" },
          )
        }
      />
      <TransitionSeries>
        {SCENE_ORDER.map((id, i) => {
          const Scene = SCENES[id];
          return (
            <React.Fragment key={id}>
              {i > 0 && (
                <TransitionSeries.Transition
                  presentation={fade()}
                  timing={linearTiming({ durationInFrames: TRANSITION_FRAMES })}
                />
              )}
              <TransitionSeries.Sequence
                name={id}
                durationInFrames={SCENE_FRAMES[id]}
              >
                <Scene />
              </TransitionSeries.Sequence>
            </React.Fragment>
          );
        })}
      </TransitionSeries>

      {SECTIONS.filter((s) => s.voice).map((s) => (
        <Sequence
          key={s.title}
          name={`Voice: ${s.title}`}
          from={sceneStart(s.scenes[0]) + VOICE_LEAD_FRAMES}
        >
          <Audio src={staticFile(`voice/${s.voice}`)} />
        </Sequence>
      ))}
    </AbsoluteFill>
  );
};
