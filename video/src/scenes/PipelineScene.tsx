import React from "react";
import { AbsoluteFill, Interactive, interpolate, useCurrentFrame, useVideoConfig } from "remotion";
import { Backdrop, Eyebrow } from "../components/Backdrop";
import { color, reveal, rise } from "../theme";

// `python -m src.pipeline all` stages, in order (src/pipeline.py).
const STAGES = ["Ingest", "Clean", "Validate", "Snapshots", "Train", "Segment", "Explain", "Recommend", "Score"];
const SERVING = ["PostgreSQL", "REST API", "Dashboard"];

export const PipelineScene: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const lines = Math.round(
    interpolate(frame, [12, 2.5 * fps], [0, 1067371], { extrapolateLeft: "clamp", extrapolateRight: "clamp" }),
  );
  // Timed to 02-data-pipeline.mp3: "One command cleans the data..." ~10 s, "The rule
  // that matters most..." ~18 s, "...the tests enforce that" ~25 s, "The models..." ~27 s.
  const stagesFrom = 10.1 * fps;
  const stageStep = Math.round(0.9 * fps);
  const cutoffFrom = 18.4 * fps;
  const testsFrom = 25.2 * fps;
  const servingFrom = 29 * fps;

  return (
    <Backdrop>
      <AbsoluteFill style={{ padding: "100px 160px" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-end" }}>
          <div style={rise(frame, 0)}>
            <Eyebrow>The data</Eyebrow>
            <Interactive.Div name="Line counter" style={{ marginTop: 10, fontSize: 132, fontWeight: 800, letterSpacing: "-0.03em" }}>
              {lines.toLocaleString("en-GB")}
            </Interactive.Div>
            <Interactive.Div name="Data source" style={{ fontSize: 38, color: color.muted }}>
              real transaction lines · UCI Online Retail II · Dec 2009 – Dec 2011
            </Interactive.Div>
          </div>
          <div style={{ display: "flex", gap: 28 }}>
            {[
              { big: "10", small: "monthly snapshots", at: 15.5 * fps },
              { big: "36,454", small: "training rows", at: 16.5 * fps },
            ].map((s) => (
              <div
                key={s.small}
                style={{
                  padding: "26px 36px",
                  borderRadius: 24,
                  background: "rgba(17,28,51,0.92)",
                  border: `2px solid ${color.line}`,
                  ...rise(frame, s.at),
                }}
              >
                <div style={{ fontSize: 72, fontWeight: 800, color: color.cyan }}>{s.big}</div>
                <div style={{ fontSize: 32, color: color.muted }}>{s.small}</div>
              </div>
            ))}
          </div>
        </div>

        <div style={{ marginTop: 64, display: "flex", alignItems: "center", gap: 14, ...rise(frame, stagesFrom - 10) }}>
          <div style={{ fontSize: 30, fontWeight: 700, color: color.muted, marginRight: 12, whiteSpace: "nowrap" }}>One command</div>
          {STAGES.map((stage, i) => {
            const on = reveal(frame, stagesFrom + i * stageStep, 12);
            return (
              <div
                key={stage}
                style={{
                  flex: 1,
                  textAlign: "center",
                  padding: "20px 0",
                  borderRadius: 16,
                  fontSize: 28,
                  fontWeight: 700,
                  border: `2px solid ${on > 0.5 ? color.indigo : color.line}`,
                  background: `rgba(99,102,241,${0.08 + 0.42 * on})`,
                  color: on > 0.5 ? "#FFFFFF" : color.muted,
                }}
              >
                {stage}
              </div>
            );
          })}
        </div>

        <div style={{ marginTop: 56, ...rise(frame, cutoffFrom) }}>
          <Interactive.Div name="Cutoff headline" style={{ fontSize: 48, fontWeight: 700 }}>
            Every feature uses only what was known before its cutoff date
          </Interactive.Div>
          <div style={{ marginTop: 72, display: "flex", alignItems: "stretch", height: 110, fontSize: 32, fontWeight: 600 }}>
            <div
              style={{
                flex: 2.4,
                borderRadius: "20px 0 0 20px",
                background: "rgba(99,102,241,0.35)",
                border: `2px solid ${color.indigo}`,
                display: "flex",
                alignItems: "center",
                paddingLeft: 36,
              }}
            >
              Features: purchase history before the cutoff
            </div>
            <div style={{ width: 8, background: color.amber, position: "relative" }}>
              <div style={{ position: "absolute", top: -54, left: -60, width: 128, textAlign: "center", color: color.amber, fontSize: 30, fontWeight: 800 }}>
                cutoff
              </div>
            </div>
            <div
              style={{
                flex: 1,
                borderRadius: "0 20px 20px 0",
                background: "rgba(34,211,238,0.18)",
                border: `2px solid ${color.cyan}`,
                display: "flex",
                alignItems: "center",
                paddingLeft: 32,
                opacity: reveal(frame, cutoffFrom + 25),
              }}
            >
              Label: the next 90 days
            </div>
          </div>
          <div
            style={{
              marginTop: 22,
              display: "inline-block",
              padding: "12px 26px",
              borderRadius: 999,
              background: "rgba(52,211,153,0.15)",
              color: color.green,
              fontSize: 32,
              fontWeight: 700,
              opacity: reveal(frame, testsFrom),
            }}
          >
            ✓ Enforced by leakage tests
          </div>
        </div>

        <div style={{ marginTop: 40, display: "flex", alignItems: "center", gap: 20, fontSize: 36, fontWeight: 700 }}>
          <div style={{ color: color.muted, opacity: reveal(frame, servingFrom) }}>Results written to</div>
          {SERVING.map((s, i) => (
            <React.Fragment key={s}>
              {i > 0 && <div style={{ color: color.cyan, opacity: reveal(frame, servingFrom + i * 12) }}>→</div>}
              <div
                style={{
                  padding: "16px 30px",
                  borderRadius: 16,
                  background: "rgba(34,211,238,0.14)",
                  border: `2px solid ${color.cyan}`,
                  ...rise(frame, servingFrom + i * 12, 20),
                }}
              >
                {s}
              </div>
            </React.Fragment>
          ))}
        </div>
      </AbsoluteFill>
    </Backdrop>
  );
};
