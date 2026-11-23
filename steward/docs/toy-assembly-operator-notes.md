# Toy airplane assembly example

This is an English usage guide. Reproducing a reported model evaluation also requires its exact task prompt and evaluation settings.

Complete only the requested airplane assembly. Seat a fastener from the authorized set in every required mating hole, without using a screwdriver (`tools.mode: "none"`), and leave the final assembly pose unchanged (`finalPose: "as_is"`). Do not manipulate or take parts from another set. Observe first, obtain entity identifiers from Ego-State, and identify the support holding each authorized fastener. An example session budget is `budgetSimS: 600`. With a 300-second program limit, wait at most 240 seconds in each program, query `skill_status`, and return before issuing the next program.

## Assembly sessions

Bind only the requested assembly goal. Use current observations to identify mating interfaces, entities, authorized materials and their observed sources. A shared category or support does not establish membership in the requested set; resolve ambiguous membership or mating relationships first.

The scene may already contain a partial assembly. Observe without moving it: stacked or overlapping parts may already form a mating interface. Before `complete_assembly`, do not pick, place, flip, separate or rearrange parts to tidy the scene. Pass each observed interface in its current state, including interfaces with seated parts or fasteners. The session determines what is complete, what still needs fastening and what has not started. Do not manually dismantle structures the session has accepted as seated.

Select `complete_assembly` with `ego_predict` and supply:

- `interfaces`: `[{part, body, hole?, heldHole?}]`, one complete mating interface per item. Selectors come from observations or the interface catalogue; omit them only when the match is unique.
- `materials`: `[{entity, source}]`, explicitly binding authorized fasteners and their observed source entities. Category wildcards are not accepted.
- `tools`: `{mode: "none" | "one" | "all", entity?}`. Select no tool action, action on one eligible hole, or action on every hole. Bind an observed tool when required. Omission means no tool action.
- `finalPose`: `"initial_up" | "as_is"`. The default `as_is` preserves the finishing pose. `initial_up` requests restoration of the body's initial upward face after all interfaces and tool actions; report insufficient budget, unstable placement or unresolved orientation.
- `budgetSimS`: the remaining simulation-time budget authorized for this session.

Run a feasible receipt, keep its returned `session`, and wait for or query the job. The session schedules interfaces, per-hole continuation and necessary flips. Tool actions follow interface releases when no explicit contrary evidence remains. Do not issue other motions, acquire a tool or start another session while it runs.

Use `decision` to determine whether intervention is needed:

- `running` / `recovering`: the session still owns execution; wait.
- `needs_input`: read `reason` and `held`. Resolve uncertain goals or membership, identity conflicts, missing authorized materials, insufficient pose or support evidence, risk refusal or exhausted bounded recovery. Observe or ask for a decision as needed; do not switch assemblies independently.
- `budget_exhausted` / `cancelled`: stop and retain the ledger. A new session must not silently reset budget or retry limits.
- `executed_unverified`: the requested actions ran, but observations do not establish the complete goal. Report uncertainty.
- `complete_observed`: current robot observations support completion. Report only that evidence, without claiming external evaluator confirmation.

After an interruption, if the sensor episode is unchanged and budget remains, predict and run `complete_assembly` with `{session: <returned reference>}`. Do not resubmit the goal or budget. An invalid identity or changed episode prevents reuse of the old session.

`released`, `confirmed` and `required` count released fasteners, observed confirmations and required holes. An unverified release may still occupy a hole, so do not add another fastener. A confirmed failure permits only recovery of the same fastener as allowed by the ledger. Report `logError` as a recording failure. Neither a low-level `succeeded` job nor historical release counts establish assembly completion.
