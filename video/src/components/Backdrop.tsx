import React from "react";
import { AbsoluteFill } from "remotion";
import { color, sans } from "../theme";

/** Dark navy backdrop with a faint grid, shared by the animated scenes. */
export const Backdrop: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <AbsoluteFill
    style={{
      backgroundColor: color.bg,
      backgroundImage:
        "radial-gradient(circle at 18% 12%, rgba(99,102,241,0.22), transparent 45%), radial-gradient(circle at 85% 90%, rgba(34,211,238,0.14), transparent 40%), linear-gradient(rgba(148,163,184,0.06) 1px, transparent 1px), linear-gradient(90deg, rgba(148,163,184,0.06) 1px, transparent 1px)",
      backgroundSize: "100% 100%, 100% 100%, 64px 64px, 64px 64px",
      fontFamily: sans,
      color: color.text,
    }}
  >
    {children}
  </AbsoluteFill>
);

/** Small uppercase label above a scene title. */
export const Eyebrow: React.FC<{ children: React.ReactNode; tone?: string }> = ({
  children,
  tone = color.cyan,
}) => (
  <div
    style={{
      fontSize: 30,
      fontWeight: 700,
      letterSpacing: "0.14em",
      textTransform: "uppercase",
      color: tone,
    }}
  >
    {children}
  </div>
);
