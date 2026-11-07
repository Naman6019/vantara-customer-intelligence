import React from "react";
import { Composition, Folder } from "remotion";
import { FPS, HEIGHT, SCENE_FRAMES, SCENE_ORDER, TOTAL_FRAMES, WIDTH } from "./timeline";
import { SCENES, Walkthrough } from "./Walkthrough";

export const RemotionRoot: React.FC = () => (
  <>
    <Composition
      id="Walkthrough"
      component={Walkthrough}
      width={WIDTH}
      height={HEIGHT}
      fps={FPS}
      durationInFrames={TOTAL_FRAMES}
    />
    <Folder name="Scenes">
      {SCENE_ORDER.map((id) => (
        <Composition
          key={id}
          id={id}
          component={SCENES[id]}
          width={WIDTH}
          height={HEIGHT}
          fps={FPS}
          durationInFrames={SCENE_FRAMES[id]}
        />
      ))}
    </Folder>
  </>
);
