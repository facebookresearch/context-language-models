# Context Language Models (CLMs)

<p align="center">
  <a href="https://rulinshao.github.io/">Rulin Shao</a><sup>1,2</sup>,
  <a href="https://www.szj.io/">Shannon Zejiang Shen</a><sup>3</sup>,
  <a href="https://oseyincs.io/">Junjie Oscar Yin</a><sup>1,2</sup>,
  <a href="https://yuetl9.github.io/">Yuetai Li</a><sup>1</sup>,
  <a href="https://minhengwang.github.io/">Minheng Wang</a><sup>1</sup>,
  <a href="https://ivison.id.au">Hamish Ivison</a><sup>1</sup>,
  <a href="https://people.ece.uw.edu/radha/">Radha Poovendran</a><sup>1</sup>,
  <a href="https://natolambert.com/">Nathan Lambert</a><sup>4</sup>,
  <a href="https://tengxiao1.github.io/">Teng Xiao</a><sup>1</sup>,
  <a href="https://ai.meta.com/people/209431298931133/mike-lewis/">Mike Lewis</a><sup>2</sup>,
  <a href="https://scottyih.org/">Wen-tau Yih</a><sup>2</sup>,
  <a href="https://homes.cs.washington.edu/~lsz/">Luke Zettlemoyer</a><sup>1,2</sup>,
  <a href="https://koh.pw/">Pang Wei Koh</a><sup>1</sup>
</p>
<p align="center">
  <sup>1</sup>University of Washington &nbsp; <sup>2</sup>Meta Superintelligence Labs &nbsp; <sup>3</sup>MIT &nbsp; <sup>4</sup>Trillium Labs
</p>
<p align="center">
  <a href="https://arxiv.org/abs/2609.37725"><img src="https://img.shields.io/badge/arXiv-2609.37725-b31b1b.svg" alt="arXiv"></a>
  <a href="https://x.com/RulinShao/status/2105282444270448647"><img src="https://img.shields.io/badge/Twitter-thread-1DA1F2.svg?logo=x&logoColor=white" alt="Twitter"></a>
</p>

<p align="center"><img src="assets/teaser.png" width="100%" alt="Context Language Models"></p>

We introduce **Context Language Models (CLMs)**, language models that natively manage their own
context. We implement this by treating the **context as a file** and allowing the model to make
unrestricted updates to this file. This allows the model to learn what is most important to
maintain in context, and naturally extends to multi-agent systems where multiple agent
contexts coexist as files.

- **Zero-shot.** Building CLMs zero-shot with existing models outperforms SOTA
  context-management strategies across a variety of tasks: 11.4% higher accuracy with 21.5%
  fewer FLOPs on BrowseComp-Plus, 5% higher scores with 59% fewer FLOPs on 12-hour EdgeBench,
  and 65% greater improvement with the same compute on a 24-hour multi-repository agent-swarm
  task.
- **In-context learning.** We show that CLMs can be steered with natural-language
  instructions evolved through a standard skill-optimization loop, improving held-out
  accuracy by up to 35.9 points on a context-management task while reducing compute.
- **Reinforcement learning.** We also introduce an online reinforcement learning method for
  CLMs, improving Qwen3.5-9B performance on BrowseComp-Plus by 47.6% while using 12% fewer
  FLOPs.

## Day 1 Support

CLM for [Pi](https://github.com/earendil-works/pi):

```sh
pi install npm:@lolipopshock/pi-clm
```

Community port for [OpenCode](https://opencode.ai),
[opencode-clm](https://github.com/bcmyguest/opencode-clm):

```sh
opencode plug opencode-clm
```

## Getting started

Run the minimal CLM agent on any [Harbor](https://github.com/laude-institute/harbor) task:

```bash
pip install -e .
clm-harbor run -p <harbor-task> -a clm-minimal -m openai/<model> \
  --agent-kwarg api_base=http://localhost:8000/v1
```

`clm-harbor` is the Harbor CLI with CLM available as `-a clm-minimal`. See
[`clm/clm_harness`](clm/clm_harness/) for configuration and serving.

## Repository

| | |
|---|---|
| [`clm/clm_harness`](clm/clm_harness/) | CLMs implemented in [Harbor](https://github.com/laude-institute/harbor) |
| [`clm/clm_icl`](clm/clm_icl/) | In-context learning for CLMs |
| [`clm/clm_rl`](clm/clm_rl/) | Reinforcement learning for CLMs |
| [`suffix_cache_reuse`](suffix_cache_reuse/) | Suffix Cache Reuse for CLM efficient serving |

## Coming soon

- [ ] ContextBench

## Citation

If you find our work helpful, we would appreciate it if you could cite our paper:

```bibtex
@article{shao2026context,
  title   = {Context Language Models},
  author  = {Shao, Rulin and Shen, Shannon Zejiang and Yin, Junjie Oscar and Li, Yuetai and
             Wang, Minheng and Ivison, Hamish and Poovendran, Radha and Lambert, Nathan and
             Xiao, Teng and Lewis, Mike and Yih, Wen-tau and Zettlemoyer, Luke and Koh, Pang Wei},
  journal = {arXiv preprint arXiv:2609.37725},
  year    = {2026}
}
```

## License

This project is licensed under [CC BY-NC 4.0](LICENSE). See also [NOTICE](NOTICE).
