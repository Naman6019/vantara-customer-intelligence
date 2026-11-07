import { Video } from "@remotion/media";
import React from "react";
import { AbsoluteFill, Easing, interpolate, staticFile, useCurrentFrame, useVideoConfig } from "remotion";
import { color, reveal, sans } from "../theme";

/**
 * A 16:9 region of the 1920×1080 recording to fill the frame with: centre (`cx`, `cy`)
 * and width `w` in recording pixels. Framing a region rather than a zoom factor keeps
 * whole UI elements in shot; see the presets below.
 */
export type Shot = { cx: number; cy: number; w: number };

/** Camera keyframe: at `t` seconds the frame shows `shot`. */
export type Zoom = Shot & { t: number };

/** The whole recording. */
export const FULL: Shot = { cx: 960, cy: 540, w: 1920 };
// The dashboard's main column spans x≈380–1840 beside a 300 px sidebar. These shots
// show that full width (x 340–1880, 1.25×), so no content is cut at the sides and the
// sidebar is left out entirely; they differ only in how far down the page they look.
export const MAIN_TOP: Shot = { cx: 1110, cy: 433, w: 1540 };
export const MAIN_MID: Shot = { cx: 1110, cy: 520, w: 1540 };
export const MAIN_LOW: Shot = { cx: 1110, cy: 647, w: 1540 };

/** On-screen note shown from `from` to `to` seconds. */
export type Callout = { from: number; to: number; title: string; body: string; top?: boolean };

const ease = Easing.bezier(0.45, 0, 0.2, 1);

const track = (t: number, zooms: Zoom[], key: keyof Shot) =>
  interpolate(
    t,
    zooms.map((z) => z.t),
    zooms.map((z) => z[key]),
    { extrapolateLeft: "clamp", extrapolateRight: "clamp", easing: ease },
  );

/** CSS transform that fills the frame with the shot, never showing past the recording. */
const camera = (t: number, zooms: Zoom[], width: number, height: number) => {
  const s = width / track(t, zooms, "w");
  const clamp = (v: number, min: number) => Math.min(0, Math.max(min, v));
  const tx = clamp(width / 2 - track(t, zooms, "cx") * s, width - width * s);
  const ty = clamp(height / 2 - track(t, zooms, "cy") * s, height - height * s);
  return `translate(${tx}px, ${ty}px) scale(${s})`;
};

/** Full-frame screen recording with eased zooms, a LIVE tag and timed callouts. */
export const Footage: React.FC<{
  file: string;
  label: string;
  zooms: Zoom[];
  callouts: Callout[];
}> = ({ file, label, zooms, callouts }) => {
  const frame = useCurrentFrame();
  const { fps, width, height } = useVideoConfig();
  const t = frame / fps;

  return (
    <AbsoluteFill style={{ backgroundColor: "#FFFFFF", fontFamily: sans, overflow: "hidden" }}>
      <AbsoluteFill style={{ transformOrigin: "0 0", transform: camera(t, zooms, width, height) }}>
        <Video src={staticFile(`footage/${file}`)} muted style={{ width: "100%", height: "100%" }} />
      </AbsoluteFill>

      <div
        style={{
          position: "absolute",
          left: 40,
          bottom: 40,
          display: "flex",
          alignItems: "center",
          gap: 14,
          padding: "14px 24px",
          borderRadius: 999,
          background: "rgba(11,18,32,0.88)",
          color: color.text,
          fontSize: 26,
          fontWeight: 600,
          boxShadow: "0 10px 30px rgba(0,0,0,0.25)",
          opacity: reveal(frame, 6),
        }}
      >
        <span style={{ width: 14, height: 14, borderRadius: 7, background: color.red }} />
        <span style={{ fontWeight: 800, letterSpacing: "0.08em" }}>LIVE</span>
        <span style={{ color: color.muted }}>{label}</span>
      </div>

      {callouts.map((c) => {
        const visible = Math.min(reveal(frame, c.from * fps, 14), 1 - reveal(frame, c.to * fps, 12));
        return (
          <div
            key={c.title}
            style={{
              position: "absolute",
              right: 60,
              ...(c.top ? { top: 60 } : { bottom: 60 }),
              width: 760,
              padding: "34px 40px",
              borderRadius: 26,
              background: "rgba(11,18,32,0.93)",
              border: `2px solid ${color.indigo}`,
              boxShadow: "0 24px 60px rgba(15,23,42,0.35)",
              color: color.text,
              opacity: visible,
              translate: `0px ${(1 - visible) * 30}px`,
            }}
          >
            <div style={{ fontSize: 46, fontWeight: 800, lineHeight: 1.12 }}>{c.title}</div>
            <div style={{ marginTop: 16, fontSize: 32, lineHeight: 1.35, color: "#CBD5E1" }}>{c.body}</div>
          </div>
        );
      })}
    </AbsoluteFill>
  );
};
