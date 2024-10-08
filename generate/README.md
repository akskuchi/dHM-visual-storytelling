For obtaining stories using existing visual storytelling models, checkout GLAC Net[^1], AREL[^2], and TAPM[^3] papers & code repositories.

[^1]: https://arxiv.org/abs/1805.10973/
[^2]: https://aclanthology.org/P18-1083/
[^3]: https://openaccess.thecvf.com/content/CVPR2021/html/Yu_Transitional_Adaptation_of_Pretrained_Models_for_Visual_Storytelling_CVPR_2021_paper/

In this work, we proposed the following approaches for visual storytelling:
- Zero-shot generation using vision-language foundation models (prompted under **visual** and **linguistic** settings&mdash;see Section 4 and Appendix A of the paper for further details):
    - BLIP-2
    - LLaVA v1.6
- Visual storytelling specific models (improvements to the TAPM model&mdash;see Section 5 and Appendix B of the paper for further details):
    - TAPM (+LLAMA 2)
    - TAPM (+ViT)


For generating stories using foundation models, use utilities in this folder. E.g.,   
`python generate/LLaVA_NeXT.py --dataset VIST --prompt P3 --sample_run`

**Note:** For BLIP-2 we used the same experimental settings as LLaVA and found that it failed to understand the instructions (e.g., for the visual context setting). Further details are available in the paper.

**Note:** To generate stories from scratch using the TAPM (or its improved versions), please [create a ticket](https://github.com/akskuchi/dHM-visual-storytelling/issues/new) and we will upload/share links to the trained model checkpoints.