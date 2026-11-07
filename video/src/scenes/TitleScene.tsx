import React from "react";
import { AbsoluteFill, Easing, Interactive, interpolate, useCurrentFrame } from "remotion";
import { Backdrop } from "../components/Backdrop";
import { rise } from "../theme";

export const TitleScene: React.FC = () => {
  const frame = useCurrentFrame();

  return (
    <Backdrop>
      <AbsoluteFill style={{ justifyContent: "center", padding: "0 180px" }}>
        <Interactive.Div
          name="Logo"
          style={{
            width: 120,
            height: 120,
            borderRadius: 28,
            background: "linear-gradient(135deg, #4F46E5, #06B6D4)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            fontSize: 72,
            fontWeight: 800,
            color: "#FFFFFF",
            boxShadow: "0 20px 60px rgba(79,70,229,0.45)",
            scale: interpolate(frame, [0, 24], [0.6, 1], {
              extrapolateLeft: "clamp",
              extrapolateRight: "clamp",
              easing: Easing.spring({ damping: 14 }),
              output: "perceptual-scale",
            }),
            opacity: interpolate(frame, [0, 10], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp" }),
          }}
        >
          V
        </Interactive.Div>
        <Interactive.Div
          name="Title"
          style={{ marginTop: 56, fontSize: 132, fontWeight: 800, lineHeight: 1.02, letterSpacing: "-0.03em", ...rise(frame, 8) }}
        >
          Vantara Customer Intelligence
        </Interactive.Div>
        <Interactive.Div
          name="Subtitle"
          style={{ marginTop: 28, fontSize: 52, fontWeight: 500, color: "#CBD5E1", ...rise(frame, 18) }}
        >
          Predicting who will leave, what they're worth, and what to do about it
        </Interactive.Div>
        <Interactive.Div
          name="Presenter"
          style={{ marginTop: 64, fontSize: 36, fontWeight: 600, color: "#94A3B8", ...rise(frame, 30) }}
        >
          Day 20 walkthrough · Naman Manocha
        </Interactive.Div>
      </AbsoluteFill>
    </Backdrop>
  );
};
