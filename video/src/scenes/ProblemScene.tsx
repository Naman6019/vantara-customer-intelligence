import React from "react";
import { AbsoluteFill, Interactive, interpolate, useCurrentFrame, useVideoConfig } from "remotion";
import { Backdrop, Eyebrow } from "../components/Backdrop";
import { color, reveal, rise } from "../theme";

// The three costs of rule-based retention named in PRD §2.1 and script §1.
const COSTS = [
  {
    n: "1",
    old: "Budget spread evenly",
    detail: "Every at-risk customer gets the same treatment, whatever they're worth",
  },
  {
    n: "2",
    old: "Valuable customers noticed late",
    detail: "Flagged only after they've already stopped buying for months",
  },
  {
    n: "3",
    old: "Best-seller suggestions",
    detail: "The same products for everyone, whatever they've bought before",
  },
];

export const ProblemScene: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  // Each card lands on its sentence in 01-problem.mp3; this scene starts 5.5 s into §1.
  const cardStart = [6.7 * fps, 10.3 * fps, 14.6 * fps];
  const turn = 17.6 * fps; // "I built a platform that replaces each of those rules..."
  const dim = interpolate(frame, [turn, turn + 20], [1, 0.35], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });

  return (
    <Backdrop>
      <AbsoluteFill style={{ padding: "110px 160px" }}>
        <div style={rise(frame, 0)}>
          <Eyebrow tone={color.amber}>The old way</Eyebrow>
          <Interactive.Div name="Headline" style={{ marginTop: 18, fontSize: 76, fontWeight: 800, letterSpacing: "-0.02em" }}>
            One rule for everyone: <span style={{ color: color.amber }}>"no order in 90 days"</span>
          </Interactive.Div>
        </div>

        <div style={{ display: "flex", gap: 40, marginTop: 64 }}>
          {COSTS.map((c, i) => (
            <div
              key={c.n}
              style={{
                flex: 1,
                padding: "44px 40px",
                borderRadius: 28,
                background: "rgba(17,28,51,0.92)",
                border: `2px solid ${color.line}`,
                ...rise(frame, cardStart[i], 60),
                opacity: reveal(frame, cardStart[i]) * dim,
              }}
            >
              <div
                style={{
                  width: 72,
                  height: 72,
                  borderRadius: 18,
                  background: "rgba(251,191,36,0.15)",
                  color: color.amber,
                  fontSize: 40,
                  fontWeight: 800,
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                }}
              >
                {c.n}
              </div>
              <div style={{ marginTop: 32, fontSize: 52, fontWeight: 700, lineHeight: 1.1 }}>{c.old}</div>
              <div style={{ marginTop: 22, fontSize: 34, lineHeight: 1.35, color: color.muted }}>{c.detail}</div>
            </div>
          ))}
        </div>

        <Interactive.Div
          name="Answer"
          style={{
            position: "absolute",
            bottom: 100,
            alignSelf: "center",
            padding: "30px 56px",
            borderRadius: 999,
            background: "linear-gradient(90deg, rgba(79,70,229,0.9), rgba(6,182,212,0.9))",
            fontSize: 54,
            fontWeight: 700,
            ...rise(frame, turn + 10, 30),
          }}
        >
          Each rule → a prediction, with a reason
        </Interactive.Div>
      </AbsoluteFill>
    </Backdrop>
  );
};
