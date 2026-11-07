import React from "react";
import { AbsoluteFill, Interactive, interpolate, useCurrentFrame, useVideoConfig } from "remotion";
import { Backdrop, Eyebrow } from "../components/Backdrop";
import { color, reveal, rise } from "../theme";
import { MUSIC } from "../timeline";

const OUTCOMES = [
  {
    title: "Retention money goes where the value is",
    proof: "£11.94 back per £ vs £2.61 for contacting the riskiest",
    at: 1,
  },
  {
    title: "Risky customers flagged early, with a reason",
    proof: "71% of churners caught · SHAP and LIME for every customer",
    at: 3.8,
  },
  {
    title: "Product suggestions are personal",
    proof: "39% bought a new suggestion vs 29% for best-sellers",
    at: 6.3,
  },
];

export const ClosingScene: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  // Timed to 07-next.mp3: "What isn't proven yet" at ~9 s, "Thanks for watching" at ~18.6 s.
  const endCard = 18 * fps;
  const summary = 1 - reveal(frame, endCard, 20);

  return (
    <Backdrop>
      <AbsoluteFill style={{ padding: "100px 160px", opacity: summary }}>
        <div style={rise(frame, 0)}>
          <Eyebrow tone={color.green}>What it does</Eyebrow>
        </div>
        <div style={{ marginTop: 30, display: "flex", flexDirection: "column", gap: 30 }}>
          {OUTCOMES.map((o) => (
            <div key={o.title} style={{ display: "flex", gap: 34, alignItems: "center", ...rise(frame, o.at * fps, 40) }}>
              <div
                style={{
                  width: 84,
                  height: 84,
                  flexShrink: 0,
                  borderRadius: 24,
                  background: "rgba(52,211,153,0.16)",
                  color: color.green,
                  fontSize: 52,
                  fontWeight: 800,
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                }}
              >
                ✓
              </div>
              <div>
                <div style={{ fontSize: 58, fontWeight: 800, lineHeight: 1.1 }}>{o.title}</div>
                <div style={{ marginTop: 8, fontSize: 36, color: color.muted }}>{o.proof}</div>
              </div>
            </div>
          ))}
        </div>

        <Interactive.Div
          name="Not proven yet"
          style={{
            marginTop: 64,
            padding: "36px 44px",
            borderRadius: 28,
            border: `3px solid ${color.amber}`,
            background: "rgba(251,191,36,0.1)",
            ...rise(frame, 9.3 * fps, 40),
          }}
        >
          <div style={{ fontSize: 48, fontWeight: 800, color: color.amber }}>Not proven yet: the real-world effect</div>
          <div style={{ marginTop: 12, fontSize: 38, color: "#E2E8F0" }}>
            Save rate and click-through are unmeasured. Next step: an A/B test.
          </div>
        </Interactive.Div>
      </AbsoluteFill>

      <AbsoluteFill style={{ justifyContent: "center", alignItems: "center", opacity: reveal(frame, endCard + 10, 24) }}>
        <Interactive.Div
          name="Thanks"
          style={{
            fontSize: 140,
            fontWeight: 800,
            letterSpacing: "-0.03em",
            scale: interpolate(frame, [endCard + 10, endCard + 40], [0.94, 1], {
              extrapolateLeft: "clamp",
              extrapolateRight: "clamp",
            }),
          }}
        >
          Thanks for watching
        </Interactive.Div>
        <Interactive.Div name="Repo" style={{ marginTop: 28, fontSize: 40, color: color.muted }}>
          github.com/Naman6019/vantara-customer-intelligence
        </Interactive.Div>
        <Interactive.Div name="Music credit" style={{ position: "absolute", bottom: 80, fontSize: 28, color: color.muted }}>
          {MUSIC.credit}
        </Interactive.Div>
      </AbsoluteFill>
    </Backdrop>
  );
};
