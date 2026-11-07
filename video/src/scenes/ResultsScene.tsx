import React from "react";
import { AbsoluteFill, Interactive, useCurrentFrame, useVideoConfig } from "remotion";
import { Backdrop, Eyebrow } from "../components/Backdrop";
import { color, reveal, rise } from "../theme";

// Out-of-time test results, docs/final_report.md §1, §3 and §10. `at` (seconds) is when
// each figure is spoken in 03-results.mp3.
const RESULTS = [
  { value: "71.3%", label: "of churners caught", note: "1,556 of 2,181 · target ≥ 70%", met: true, at: 16.5 },
  { value: "0.756", label: "churn ROC-AUC", note: "target 0.80 · missed", met: false, at: 26 },
  { value: "58%", label: "of next-quarter value in the top 10%", note: "value model ranking", met: true, at: 32.5 },
  { value: "0.49", label: "value R² on the test period", note: "target 0.60 · missed", met: false, at: 38.2 },
];

const SPLITS = [
  { label: "Train", detail: "10 monthly cutoffs · Jun 2010 – Mar 2011", flex: 3, tint: "rgba(99,102,241,0.35)", edge: color.indigo },
  { label: "Validate", detail: "Jun 2011", flex: 1, tint: "rgba(148,163,184,0.18)", edge: color.muted },
  { label: "Test once", detail: "Sep 2011", flex: 1, tint: "rgba(34,211,238,0.2)", edge: color.cyan },
];

export const ResultsScene: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  return (
    <Backdrop>
      <AbsoluteFill style={{ padding: "100px 160px" }}>
        <div style={rise(frame, 0)}>
          <Eyebrow>Results</Eyebrow>
          <Interactive.Div name="Headline" style={{ marginTop: 14, fontSize: 84, fontWeight: 800, letterSpacing: "-0.02em" }}>
            Tested out-of-time: on a period the model never saw
          </Interactive.Div>
        </div>

        <div style={{ marginTop: 44, display: "flex", gap: 10, ...rise(frame, 3 * fps) }}>
          {SPLITS.map((s) => (
            <div
              key={s.label}
              style={{
                flex: s.flex,
                padding: "18px 26px",
                borderRadius: 16,
                background: s.tint,
                border: `2px solid ${s.edge}`,
              }}
            >
              <div style={{ fontSize: 34, fontWeight: 700 }}>{s.label}</div>
              <div style={{ fontSize: 26, color: "#CBD5E1" }}>{s.detail}</div>
            </div>
          ))}
        </div>

        <div style={{ marginTop: 56, display: "grid", gridTemplateColumns: "1fr 1fr 1fr 1fr", gap: 30 }}>
          {RESULTS.map((r) => {
            const tone = r.met ? color.green : color.amber;
            return (
              <div
                key={r.label}
                style={{
                  padding: "40px 34px",
                  borderRadius: 28,
                  background: "rgba(17,28,51,0.94)",
                  border: `3px solid ${tone}`,
                  ...rise(frame, r.at * fps, 60),
                }}
              >
                <div style={{ fontSize: 108, fontWeight: 800, color: tone, letterSpacing: "-0.03em" }}>{r.value}</div>
                <div style={{ marginTop: 10, fontSize: 36, fontWeight: 600, lineHeight: 1.2, minHeight: 88 }}>{r.label}</div>
                <div style={{ marginTop: 20, fontSize: 28, color: color.muted }}>
                  {r.met ? "✓ " : "△ "}
                  {r.note}
                </div>
              </div>
            );
          })}
        </div>

        <Interactive.Div
          name="Honesty note"
          style={{
            marginTop: 44,
            fontSize: 38,
            fontWeight: 600,
            color: "#CBD5E1",
            opacity: reveal(frame, 44.5 * fps),
          }}
        >
          Both misses are explained in the final report (§10)
        </Interactive.Div>
      </AbsoluteFill>
    </Backdrop>
  );
};
