# Vantara walkthrough: narration script (draft 2)

**Target length:** about 5 minutes: roughly 600 spoken words (about 4¼ minutes at 140 words
per minute), plus pauses while the footage plays.
**Message:** the platform solves the problem it was built for, and does it efficiently.

Edit freely so it sounds like you. Every number below comes from the final report or from
measurements taken on the running app; if you change one, tell me so the on-screen
graphics match.

**Recording:** one audio file per section (`01-problem.wav` … `07-next.wav`), in a quiet room,
mic about a hand-width from your mouth. Leave a second of silence at the start and end of
each file. Retakes only need to redo that one section.

---

## 1. The problem (0:00 – 0:35)

**On screen:** title card, then three "old way" cards appearing one by one.

> Vantara's retention team decides who to contact using simple rules, like "no order in
> ninety days". That has three costs. Marketing money is spread evenly instead of where it
> matters. Valuable customers are noticed only after they've already gone quiet. And product
> suggestions are just best-sellers.
>
> I built a platform that replaces each of those rules with a prediction, and every
> prediction comes with a reason.

## 2. Data and pipeline (0:35 – 1:15)

**On screen:** architecture diagram; pipeline stages light up in order; counter to 1,067,371.

> It starts from the UCI Online Retail II dataset: just over a million real transaction lines
> from a UK online retailer, covering two years. One command cleans the data, checks it against data contracts, and
> builds ten monthly customer snapshots, giving more than thirty-six thousand training rows.
>
> The rule that matters most is that every feature uses only what was known before its
> cutoff date, and the tests enforce that. The models, segments, explanations and
> recommendations are then written to a database that the API and dashboard read from.

## 3. Results, honestly (1:15 – 2:10)

**On screen:** four result cards (met targets in green, missed in amber); confusion matrix.

> I evaluated everything out-of-time: train on the past, then test once on a later period
> the model has never seen. That's harder than a random split, and it shows in the numbers.
>
> At its operating threshold, the churn model catches seventy-one percent of customers who
> go on to leave: one thousand five hundred and fifty-six out of two thousand one hundred and
> eighty-one. That meets the seventy percent recall target. Its AUC is 0.76, below the 0.80
> target.
>
> The value model is good at ranking: the top tenth of customers it picks hold fifty-eight
> percent of the next quarter's value. But its R-squared on the test period is 0.49, short of
> the 0.60 target. Both misses are explained in the final report.

## 4. Where the money goes (2:10 – 3:00) · live footage

**On screen:** At-Risk Priority page, then the Campaign Planner.

> Here's the live dashboard, reading everything from the API. The at-risk list ranks
> customers by value at risk: the chance they leave, multiplied by what they're likely to
> spend in the next ninety days. That's about four hundred and thirteen thousand pounds at risk across four thousand
> two hundred and sixty-nine customers, and the list puts the most valuable first, not just
> the most likely to leave.
>
> The campaign planner takes a budget, here five hundred pounds, and picks who to contact to
> get the most expected value back per pound. Compared on the same budget, targeting by value
> at risk returns more than four times as much per pound as simply contacting the customers
> most likely to leave.
>
> Those pound figures rest on an assumed save rate, because there's no experiment data yet,
> so the planner also shows how the answer changes across a range of save rates.

## 5. One customer, explained (3:00 – 3:50) · live footage

**On screen:** Customer 360 for one customer; SHAP and LIME charts; recommended products.

> Open any customer and you can see why. Here's their churn risk, predicted value, segment and
> recommended action. SHAP and LIME show which behaviours drive the score, calculated live by
> the API.
>
> Below that are product suggestions. They're personalised from purchase history, and each
> one names the past purchase it comes from. Tested on later purchases, thirty-nine percent of
> customers bought at least one of their ten suggestions that they had not bought before,
> compared with twenty-nine percent for best-sellers.

## 6. Fast, and easy to run (3:50 – 4:25) · live footage

**On screen:** Swagger docs, a live `/predict` call; terminal running `docker compose up`.

> The same predictions are available through a documented REST API. A customer lookup takes
> about six milliseconds at the ninety-fifth percentile, and a full prediction under sixty,
> well inside the four-hundred-millisecond target.
>
> The whole stack starts with one command, and a hundred and seventy-three automated tests
> cover more than eighty percent of the code.

## 7. What's next (4:25 – 5:00)

**On screen:** closing card with the three outcomes; "next: A/B test".

> So, retention money goes where the value is, risky customers are flagged early with a
> reason, and product suggestions are personal.
>
> What isn't proven yet is the real-world effect. The save rate and click-through haven't
> been measured, and the next step is an A/B test to measure both. Thanks for watching.
