import React from "react";
import { FULL, Footage, MAIN_MID, MAIN_LOW, MAIN_TOP, Shot } from "../components/Footage";

// Swagger's column is centred (x≈250–1670), unlike the dashboard's.
const API_CENTRE: Shot = { cx: 960, cy: 600, w: 1540 };

// Zoom and callout timings are seconds into each clip; see capture/capture.mjs for
// what happens when. Figures come from docs/final_report.md and the live API.

/** §4: executive cockpit, then the value-at-risk leaderboard. */
export const AtRiskScene: React.FC = () => (
  <Footage
    file="at_risk.mp4"
    label="Dashboard · Docker"
    zooms={[
      { t: 0, ...FULL },
      { t: 0.8, ...MAIN_TOP },
      { t: 3.6, ...MAIN_TOP },
      { t: 4.6, ...FULL },
      { t: 6.5, ...FULL },
      { t: 7.6, ...MAIN_MID },
      { t: 20.5, ...MAIN_MID },
      { t: 22.2, ...FULL },
    ]}
    callouts={[
      {
        from: 5.5,
        to: 13,
        title: "Ranked by value at risk",
        body: "Chance of leaving × predicted spend in the next 90 days, so the most valuable come first",
      },
      {
        from: 13.3,
        to: 21,
        title: "£412,835 at risk",
        body: "Across 4,269 scored customers",
      },
    ]}
  />
);

/** §4: campaign planner at a £500 budget. */
export const PlannerScene: React.FC = () => (
  <Footage
    file="planner.mp4"
    label="Campaign planner · Docker"
    zooms={[
      { t: 0, ...FULL },
      { t: 6.2, ...FULL },
      { t: 7.2, ...MAIN_TOP },
      { t: 18.5, ...MAIN_TOP },
      { t: 19.5, ...FULL },
    ]}
    callouts={[
      {
        from: 10,
        to: 18.5,
        title: "£11.94 vs £2.61 back per £",
        body: "Targeting by value at risk vs contacting those most likely to leave · £500 budget",
      },
      {
        from: 19.5,
        to: 28.5,
        title: "The save rate is an assumption",
        body: "No experiment data yet, so the planner shows the answer at 5%, 10% and 20%",
      },
    ]}
  />
);

/** §5: one high-risk customer, from action to suggestions to SHAP and LIME. */
export const Customer360Scene: React.FC = () => (
  <Footage
    file="customer360.mp4"
    label="Customer 360 · Docker"
    zooms={[
      { t: 0, ...FULL },
      { t: 3.6, ...FULL },
      { t: 4.6, ...MAIN_TOP },
      { t: 30.6, ...MAIN_TOP },
      { t: 32.6, ...MAIN_LOW },
    ]}
    callouts={[
      {
        from: 4.6,
        to: 13.2,
        title: "84% churn risk",
        body: "No order for 274 days → automated win-back email",
      },
      {
        from: 15.6,
        to: 30.4,
        title: "Suggestions with a reason",
        body: "39% of test customers bought at least one new suggestion · best-sellers: 29%",
      },
      {
        from: 32.6,
        to: 42.4,
        title: "SHAP and LIME agree",
        body: "Low engagement and a long gap since the last order drive this customer's risk",
        top: true,
      },
    ]}
  />
);

/** §6: a live POST /predict in the API's Swagger docs. */
export const ApiPredictScene: React.FC = () => (
  <Footage
    file="api_predict.mp4"
    label="REST API · Docker"
    zooms={[
      { t: 0, ...FULL },
      { t: 9.5, ...FULL },
      { t: 11, ...API_CENTRE },
      { t: 17.5, ...API_CENTRE },
    ]}
    callouts={[
      {
        from: 1,
        to: 6.2,
        title: "Documented REST API",
        body: "FastAPI with OpenAPI docs; the dashboard reads the same endpoints",
      },
      {
        from: 6.6,
        to: 16.8,
        title: "Well inside the 400 ms target",
        body: "p95: about 6 ms for a customer lookup, under 60 ms for a full prediction",
      },
    ]}
  />
);
