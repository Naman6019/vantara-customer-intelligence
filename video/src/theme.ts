import { loadFont as loadInter } from "@remotion/google-fonts/Inter";
import { loadFont as loadMono } from "@remotion/google-fonts/JetBrainsMono";
import { Easing, interpolate } from "remotion";

export const { fontFamily: sans } = loadInter("normal", {
  weights: ["400", "500", "600", "700", "800"],
  subsets: ["latin"],
});
export const { fontFamily: mono } = loadMono("normal", {
  weights: ["400", "600"],
  subsets: ["latin"],
});

// The dashboard's own palette (frontend/theme.py), so animated scenes and footage match.
export const color = {
  bg: "#0B1220",
  panel: "#111C33",
  line: "#22304D",
  text: "#F1F5F9",
  muted: "#94A3B8",
  indigo: "#6366F1",
  cyan: "#22D3EE",
  green: "#34D399",
  amber: "#FBBF24",
  red: "#F87171",
};

const easeOut = Easing.bezier(0.16, 1, 0.3, 1);

/** 0 -> 1 over `duration` frames starting at `start`, eased. */
export const reveal = (frame: number, start: number, duration = 18): number =>
  interpolate(frame, [start, start + duration], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: easeOut,
  });

/** Opacity and lift for an element that rises into place at `start`. */
export const rise = (frame: number, start: number, distance = 40) => {
  const p = reveal(frame, start);
  return { opacity: p, translate: `0px ${(1 - p) * distance}px` };
};
