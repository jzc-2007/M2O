# Third-party notices

Original M2O additions are MIT licensed. Existing copyright headers remain in
place. The root license does not relicense third-party code, model weights,
evaluation statistics, manuscript text, or generated research images.

- [iMeanFlow](https://github.com/Lyy-iiis/imeanflow), Yiyang Lu, MIT. Runtime
  checkout pinned to `bf60cd7cb653f6628e59d48034b333c5eba445e2`; downloaded
  separately. License: `third_party/iMeanFlow-LICENSE`.
- [BiFlow](https://github.com/Lyy-iiis/BiFlow), Yiyang Lu, MIT. The ConvNeXt JAX
  adapter and inherited evaluation utilities were adapted from this research
  code lineage. License: `third_party/BiFlow-LICENSE`.
- [ConvNeXt V2](https://github.com/facebookresearch/ConvNeXt-V2), Meta Platforms,
  MIT for code; pretrained/fine-tuned ImageNet weights are CC BY-NC 4.0. The
  upstream combined license is retained in `third_party/ConvNeXt-V2-LICENSE`.
  The public `facebook/convnextv2-tiny-22k-224` checkpoint is downloaded, not
  bundled into this source repository.
- Inception evaluation includes ports/adaptations of
  [TorchVision](https://github.com/pytorch/vision) (BSD),
  [PyTorch-FID](https://github.com/mseitzer/pytorch-fid) (Apache 2.0), and
  [Flax](https://github.com/google/flax) (Apache 2.0).
  Corresponding license texts are in `third_party/`.
- VAE loading uses Hugging Face Diffusers (Apache 2.0). The separately downloaded
  `enterprise-explorers/sd-vae-ft-mse-flax` checkpoint is a conversion of the
  Stable Diffusion VAE; consult its model card and the original model terms.
- ICLR LaTeX style and bundled natbib/fancyhdr files retain their own notices.

This release provides research adaptation code; it does not grant broader
commercial rights to pretrained assets. Asset provenance, revisions and
checksums are recorded in `scripts/assets.lock.json`.
