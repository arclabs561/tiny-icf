#set page(margin: 2cm)
#set text(size: 11pt)
#set heading(numbering: "1.")

// Code highlighting available via codly package if installed
// #import "@preview/codly:0.1.0": *
// #show: codly.with(theme: "github-dark")

#title[
  Design Philosophy: Learning Through Experimentation
]

#text(fill: gray)[
  ICF Estimation Project · #datetime.today().display()
]

#par(justify: true)[
  This document captures the design philosophy and approach that guides this project - a focus on learning, experimentation, and understanding rather than perfect metrics or production optimization.
]

== The Core Insight

#strong[This is a fun experimental project, not a production system.]

The goal is to #strong[learn, experiment, and build something interesting] - not to achieve perfect metrics or production-grade performance.

== What Makes a Good Experimental Project?

=== 1. Learning & Understanding (Most Important)

- Understand how the model learns from bytes
- Discover what patterns it finds
- Explain why it works (or doesn't)
- Learn about language structure

#strong[Examples of what we've learned:]

- "The model learns that 'ing' endings are common"
- "Multi-loss training improves ranking but slows convergence"
- "Character patterns capture morphological structure even without explicit linguistic knowledge"

=== 2. Experimentation (Very Important)

- Try different architectures
- Try different loss functions
- Compare approaches
- See what happens

#strong[Examples of experiments:]

- "What if we use hierarchical embeddings?"
- "Does contrastive loss help?"
- "How does box embedding compare to standard?"

=== 3. Interesting Results (Nice to Have)

- Model does something unexpected
- Discover interesting patterns
- Find surprising behaviors

#strong[Examples of interesting results:]

- "Model predicts high ICF for portmanteaus even if not in training"
- "The model recognizes 'flimjam' as English-like despite never seeing it"
- "Multi-loss helps ranking but hurts absolute accuracy"

=== 4. Reasonable Performance (Good Enough)

- Model learns something (not just mean)
- Generalizes to unseen words
- Fast and small

#strong[Not Required:]

- Perfect accuracy
- Beating benchmarks
- Production-grade performance

== What We Focus On

=== Do This

1. #strong[Experiment with architectures]
   - Try hierarchical embeddings
   - Try box embeddings
   - Compare sizes vs performance

2. #strong[Experiment with training]
   - Try multi-loss
   - Try curriculum learning
   - See what works

3. #strong[Understand the model]
   - What patterns does it learn?
   - Why does it work?
   - What are its limitations?

4. #strong[Have fun]
   - Try weird things
   - See what happens
   - Learn something new

=== Don't Worry About

1. #strong[Perfect metrics]
   - MAE doesn't need to be < 0.1
   - Spearman doesn't need to be > 0.8
   - Good enough is good enough

2. #strong[Exhaustive optimization]
   - Don't need to try every combination
   - Don't need perfect hyperparameters
   - Don't need to squeeze every % of accuracy

3. #strong[Production deployment]
   - Not building for production
   - Not optimizing for scale
   - Not worrying about edge cases

4. #strong[Beating state-of-the-art]
   - Not competing with large models
   - Not trying to be best-in-class
   - Just trying to learn

== Realistic Expectations

=== What's Achievable (Research-Based)

For less than 50k parameter models:
- 50-70% of baseline accuracy
- MAE 0.1-0.3 (high-freq) / 0.3-0.5 (full vocab)
- Spearman 0.6-0.8 (theoretical bound ~0.18-0.19 for character-level)

Our Targets (Realistic):
- MAE < 0.25 (high-freq) / < 0.5 (full vocab) ✓
- Spearman > 0.15 (approaching theoretical bound) ✓
- Jabberwocky 3/5+ tests pass ✓
- Fast and small ✓

=== What's Not Realistic

- MAE < 0.1 (too strict for tiny models)
- Spearman > 0.8 (theoretically impossible for character-level)
- Perfect accuracy (impossible with less than 50k params)
- Production-grade robustness (not the goal)

== The Experimental Manifesto

1. #strong[Learning > Perfection]
   - Understand how it works
   - Learn from failures
   - Document insights

2. #strong[Experimentation > Optimization]
   - Try interesting things
   - See what happens
   - Don't over-optimize

3. #strong[Interesting Results > Perfect Metrics]
   - Celebrate discoveries
   - Learn from surprises
   - Share what you learned

4. #strong[Fun > Production]
   - This is for fun
   - Not for deployment
   - Enjoy the process

== What Success Looks Like

=== Success = Learning Something Interesting

#strong[Examples:]

- "I learned how CNNs extract morphological patterns from bytes"
- "Box embeddings are cool but don't help much for this task"
- "Multi-loss training improves ranking but convergence is slower"
- "The model generalizes to portmanteaus even without training data"
- "The model recognizes 'flimjam' as English-like despite never seeing it"

=== Success ≠ Perfect Metrics

#strong[Not success:]

- "Achieved MAE < 0.1" (too strict)
- "Beat all benchmarks" (not the goal)
- "Production-ready" (not needed)

== The Bottom Line

#strong[This is a fun project.] The goal is to learn, experiment, and build something interesting - not to achieve perfect metrics or production-grade performance.

#strong[Focus on:]

- Understanding how it works
- Trying interesting things
- Learning from experiments
- Having fun

#strong[Don't focus on:]

- Perfect accuracy
- Exhaustive optimization
- Production concerns
- Benchmark scores

#strong[Remember:] The best experimental projects teach you something new and leave you with interesting insights, not perfect metrics.

