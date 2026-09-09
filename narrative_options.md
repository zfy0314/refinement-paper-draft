# Chosen narrative: iterative refinement through semantic rollout analysis

## Working title and thesis

**Iterative Policy Refinement through Semantic Rollout Analysis**

A structured policy serves two complementary purposes: it provides an inductive bias for learning from demonstrations, and it exposes meaningful intermediate computations that can be analyzed to guide structural revision. The paper studies an iterative process that alternates parameter fitting with LLM-guided revision, using semantic rollout analysis to connect execution behavior to possible program edits.

**Central question:** How can the semantics exposed by a trainable policy connect execution feedback to iterative structural refinement?

## Scope and relationship to KIM

The main source is the IJCAI26 manuscript, *Automatic Closed-Loop Structured Policy Generation with LLM-Guided Semantic Reasoning for Embodied Agents*. Its refinement procedure, diagnostic interface, prompts, and Car Racing/Door Opening experiments supply the paper's content.

KIM is background only: it supplies the structured-policy representation and the separation between program structure and learned numerical parameters. Introduce it briefly and cite it as prior work. Do not reproduce its experiments, develop its representation as a new contribution, or combine its results with the refinement study as if their protocols were matched.

The project motivation is to reduce the expert's burden of explicitly describing the strategy underlying demonstrations. LLM knowledge can supply an initial structure, but that structure may be inconsistent with the demonstrated behavior. The paper develops the broader learning question of how to revise an initial structure through execution feedback. The project history motivates that question without determining the whole exposition.

## Narrative progression

1. **Structure helps learning.** With limited demonstrations, task-specific intermediate computations and dependencies shape what is easy to learn. Numerical parameters remain trainable.
2. **Initial structure is provisional.** Expert descriptions can be incomplete, and LLM knowledge may differ from the strategy expressed in demonstrations. Parameter fitting can leave problematic closed-loop behavior that motivates revising computations or constraints. This does not imply that the initial program is incapable of solving the task.
3. **Refinement needs an interface.** Named variables, their definitions, and explicit dependencies link intended behavior to numerical execution and editable computations.
4. **Semantic analysis closes the loop.** Train the current policy; record its execution; generate and run diagnostic analyses; use their results to propose a revised program; train and execute the revised program again.
5. **Evaluate the iterative process.** Establish improvements over the initial structure, compare feedback procedures, inspect actual revisions, and report downstream RL as an additional benefit.

The initial LLM input is the task description and semantic feature/action specification. Demonstrations fit parameters; rollouts of the fitted policy inform revisions. Avoid suggesting that the LLM initially reads the demonstration to infer its structure directly.

## Contribution emphasis

- An iterative procedure combining demonstration-based parameter learning with LLM-guided revision of a semantic policy program.
- A diagnostic interface that uses named internal execution traces and generated analysis programs to inform those revisions.
- An empirical study of refinement, feedback type, and downstream learning on two simulated control tasks.

Semantics enables the connection between learning and revision. The central object of study is the repeated refinement process; diagnostic scripts and token accounting support that story.

## Eight-page main-text outline

These budgets apply to the eventual manuscript, including figures. The current LaTeX contains planning bullets, not a finished eight-page paper.

| Section | Pages | Subsections/content |
|---|---:|---|
| 1. Introduction | 1.0 | Five paragraphs following the narrative progression; introduce the debugging analogy briefly. Includes abstract/title space. |
| 2. Related work | 0.5 | 2.1 Structured policies and learning from demonstrations; 2.2 LLM-guided program revision and execution feedback, including the relationship to debugging. Retain these two subsections. |
| 3. Semantic policy structure | 1.0 | 3.1 Policy programs and learnable parameters; 3.2 An interface between execution and revision. Begin the running example with a sample topology in 3.1; KIM is concise background. |
| 4. Iterative refinement through semantic analysis | 2.5 | 4.1 Refinement loop and running example; 4.2 Fitting each policy to demonstrations; 4.3 Recording semantic execution traces; 4.4 Generating and executing diagnostics; 4.5 Revising the program and continuing the loop; 4.6 Post-refinement reinforcement learning. Continue the Section 3 example through 4.1–4.5: textual parameter fitting in 4.2 and illustrated latent traces in 4.3. |
| 5. Experiments and analysis | 2.5 | 5.1 Tasks, baselines, and evaluation protocol; 5.2 Does iterative refinement improve the initial policy?; 5.3 How does the feedback affect refinement?; 5.4 What changes across refinement iterations?; 5.5 Does refinement help subsequent reinforcement learning? |
| 6. Discussion and conclusion | 0.5 | 6.1 Scope and limitations; 6.2 Takeaway. |
| **Total** | **8.0** | References and appendices excluded. |

Appendices hold generation/environment prompts, diagnostic/revision prompts, training and RL details, complete refinement records, and extended feedback/semantic analyses. All concepts required to understand the refinement loop remain in the main text.

## Debugging analogy and running example

Use **debugging a learned policy through its execution traces** as an explanatory analogy for the refinement loop. It connects familiar activities—instrumenting a program, writing diagnostic checks, inspecting outputs, editing code, and rerunning—to the actual procedure. Semantic structure is what makes the internal traces meaningful enough to support those checks and candidate edits.

| Debugging activity | Correspondence in this method |
|---|---|
| Inspect a program and its intermediate computations | Inspect the policy's named quantities, operators, and dependencies. |
| Instrument execution | Record observations, parameters, intermediate quantities, and actions during rollouts. |
| Write diagnostic checks | Generate analysis scripts expressing hypotheses about the policy and its behavior. |
| Run checks and inspect evidence | Execute scripts on complete rollout tables and return numerical summaries to the LLM. |
| Edit and rerun | Revise the policy program, fit its parameters again, and execute the newly learned policy. |

The analogy clarifies the procedure, but diagnostic results need not be binary or have a known correct answer. Some checks examine numerical consistency; others summarize correlated, continuous, potentially stochastic behavior. An expectation about driving can be incomplete or wrong. Poor behavior may reflect an unsuitable inductive bias or parameter fit rather than a software bug. The LLM's proposed edit is a hypothesis whose resulting behavior is observed after retraining. Do not imply formal verification, a fixed regression suite, or an automatic test-passing gate that the implementation does not provide.

Introduce the analogy briefly in the Introduction, relate it to nearby work in Section 2, explain instrumentation through the representation in Section 3, and make it concrete in the Section 4 method overview. Keep it as an explanatory device rather than a separate contribution or an unsupported claim that this is a general code-repair system.

### Running example from topology to refinement

Use one Car Racing steering example, beginning with a sample structured-policy topology in Section 3.1 and continuing through the method. Readers should understand the observations, semantic latents, operations, parameterized dependencies, and actions before encountering the refinement loop. The existing source's steering-alignment diagnostic is a concrete candidate: it measures the fraction of time steps for which steering and the signed lateral quantity have the expected sign relationship, with an off-track subset. Its validity depends on the coordinate convention and driving context; it is not a complete specification of correct steering during turns.

| Location | Example content and presentation |
|---|---|
| 3.1 Policy representation | **Panel A — Topology:** introduce the selected policy graph, distinguish structure from trainable numerical parameters, and explain how an action is computed. A compact code snippet may clarify correspondence. |
| 3.2 Semantic interface | Refer to the same topology and identify internal quantities whose values can be tracked during execution. |
| 4.1 Overview | Refer back to panel A and explain how the same policy enters the refinement loop; no second introductory topology panel. |
| 4.2 Parameter fitting | **Text only:** explain that LLM-initialized values may not reflect the environment's quantitative relationships; fitting adjusts parameters to increase demonstration likelihood while preserving topology. No dedicated visual evidence is needed here. |
| 4.3 Semantic traces | **Panel B — Illustrated latent behavior:** show selected semantic latent traces, observations, and actions aligned with an annotated rollout. Link plotted quantities to panel A and highlight a documented problematic interval. Avoid a large numerical table. |
| 4.4 Diagnostics | **Panel C — Check and result:** show a short generated analysis, the actual numerical result, and its interpretation. Distinguish a diagnostic indication from a guaranteed failure condition. |
| 4.5 Revision and refitting | **Panel D — Edit and subsequent behavior:** show the actual before/after computation, the rationale for the proposed edit, and the rollout after fitting the revised policy. |

Use consistent names throughout the panels. The Section 4.3 figures help the reader see how latent computations evolve; they do not change the tabular analysis interface or introduce visual feedback to the LLM. Raw log excerpts may go in the appendix. Full numerical logs go to generated analysis scripts; the LLM receives the code/schema for diagnostic generation and the resulting summaries for revision. The overview figure should distinguish the LLM, learner, environment, and diagnostic executor, with a visible return from revision to fitting.

The source snippets are not yet established as one historical sequence. Recover matched programs, logs, diagnostics, and evaluations before presenting the example as empirical evidence. Do not attach unrelated slip-weight outputs or constraint edits to the steering test. If needed, label an illustrative reconstruction schematic and keep it separate from measured results. Section 5.4 extends the example to evidence across iterations, including ineffective revisions; Door Opening remains a second task and progression.

## Retained reinforcement-learning component

RL remains in the paper. Section 4.6 briefly describes the post-refinement phase: hold the final program structure fixed and further optimize its parameters using environment rewards. Identify the original critic-free, GSPO-based recipe and its role; detailed objective, grouping assumptions, and hyperparameters remain in Appendix C.2 after verification against implementation.

Section 5.5 retains the reported RL performance and environment-reset comparison as a downstream benefit of refinement. RL does not replace the imitation-learning step within each refinement round. Allocate approximately 0.2 of the method's 2.5 pages to its main-text description so the paper remains centered on iterative refinement.

## Evidence and drafting boundaries

- Treat semantic labels as intended meanings and graph edges as computational dependencies; avoid claims of verified causality or recovery of the demonstrator's unique internal strategy.
- Distinguish prompt requests from enforced properties. Do not introduce an undocumented graph validator, convergence guarantee, or candidate acceptance/reversion mechanism.
- Record initialization and action-distribution edits alongside structural edits; the study is not a topology-only ablation.
- Use the existing feedback comparisons without claiming an isolated effect of semantic naming or a reward-only ablation that was not reported.
- Audit the IL objective sign, stopping/initialization rules, uncertainty definitions, and resource accounting before drafting numerical claims. Environment resets are not total compute.
- Keep the present task at the outline stage. The [content map](content_mapping.md) identifies reusable source material, missing artifacts, and content to create. Detailed LaTeX bullets specify what belongs in each subsection.
