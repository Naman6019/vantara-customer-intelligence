import React from "react";
import { AbsoluteFill, Interactive, useCurrentFrame } from "remotion";
import { Backdrop, Eyebrow } from "../components/Backdrop";
import { color, mono, reveal, rise } from "../theme";

// Real output, copied from public/terminal/compose_up.txt and pytest.txt.
const COMPOSE_CMD = "docker compose up --build -d";
const COMPOSE_OUT = [
  "Container vantara_postgres Healthy",
  "Container vantara_scorer Started",
  "Container vantara_scorer Exited",
  "Container vantara_api Started",
  "Container vantara_dashboard Started",
];
const PYTEST_CMD = "pytest -q --cov=src --cov=api";
const PYTEST_OUT = [
  { text: "TOTAL                              2867    557    81%", tone: color.text },
  { text: "173 passed, 8 warnings in 45.96s", tone: color.green },
];

const typed = (frame: number, start: number, text: string) =>
  text.slice(0, Math.max(0, frame - start));

// Fast enough that "173 passed" lands as 06-speed.mp3 says "a hundred and seventy-three
// automated tests", about 3.5 s into this scene.
const LINE_STEP = 4;

export const TerminalScene: React.FC = () => {
  const frame = useCurrentFrame();
  const composeDone = 10 + COMPOSE_CMD.length;
  const pytestStart = composeDone + COMPOSE_OUT.length * LINE_STEP + 10;
  const pytestDone = pytestStart + PYTEST_CMD.length;

  return (
    <Backdrop>
      <AbsoluteFill style={{ padding: "100px 160px" }}>
        <div style={rise(frame, 0)}>
          <Eyebrow>Easy to run</Eyebrow>
          <Interactive.Div name="Headline" style={{ marginTop: 14, fontSize: 76, fontWeight: 800, letterSpacing: "-0.02em", whiteSpace: "nowrap" }}>
            One command to start · 173 tests to check
          </Interactive.Div>
        </div>

        <div
          style={{
            marginTop: 56,
            paddingBottom: 18,
            borderRadius: 24,
            background: "#060A14",
            border: `2px solid ${color.line}`,
            boxShadow: "0 30px 80px rgba(0,0,0,0.45)",
            overflow: "hidden",
            ...rise(frame, 4),
          }}
        >
          <div style={{ display: "flex", gap: 12, padding: "20px 26px", borderBottom: `2px solid ${color.line}` }}>
            {["#F87171", "#FBBF24", "#34D399"].map((c) => (
              <div key={c} style={{ width: 18, height: 18, borderRadius: 9, background: c }} />
            ))}
          </div>
          <div style={{ padding: "26px 40px", fontFamily: mono, fontSize: 32, lineHeight: 1.5 }}>
            <div>
              <span style={{ color: color.cyan }}>$ </span>
              {typed(frame, 10, COMPOSE_CMD)}
            </div>
            {COMPOSE_OUT.map((line, i) => (
              <div key={line} style={{ color: color.muted, opacity: reveal(frame, composeDone + i * LINE_STEP, 4) }}>
                {line}
              </div>
            ))}
            <div style={{ marginTop: 18, opacity: reveal(frame, pytestStart - 4, 4) }}>
              <span style={{ color: color.cyan }}>$ </span>
              {typed(frame, pytestStart, PYTEST_CMD)}
            </div>
            {PYTEST_OUT.map((line, i) => (
              <div
                key={line.text}
                style={{ color: line.tone, fontWeight: 600, whiteSpace: "pre", opacity: reveal(frame, pytestDone + 8 + i * 8, 4) }}
              >
                {line.text}
              </div>
            ))}
          </div>
        </div>
      </AbsoluteFill>
    </Backdrop>
  );
};
