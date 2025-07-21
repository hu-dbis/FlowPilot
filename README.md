# FlowPilot: A Recommender System for Designing Scientific Workflows

## Abstract:

Scientific Workflows (SWF) encapsulate data processing tasks by
organizing various tools, operators, and data in a logical flow.
Due to their complex and domain-specific nature, developing SWFs remains laborious.
Current code-generating large language models (Code LLMs) struggle to assist users in developing these workflows. This limitation arises primarily from the insufficient availability of relevant training data in public repositories, making it challenging for LLMs to learn specialized patterns and domain-specific logic.

To address this, we propose FlowPilot, a recommender system for developing SWFs that
assists developers by suggesting the next operator. Our system complements Code
LLMs by enriching the source code to generate more accurate results.
FlowPilot leverages a similarity knowledge base (SKB) that indexes historical
workflows to find the ones matching the current context.
To generate relevant recommendations, FlowPilot employs a statistical approach based on Markov chains to identify the most likely next step.
As a proof of concept, we evaluated our system on NextFlow workflows and the results demonstrate the effectiveness of FlowPilot by outperforming state-of-the-art code-generating models, e.g., Llama-$4$, and traditional methods, e.g., association rule mining techniques.
