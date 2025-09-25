---
tags:
  - "#project"
---

| Reference                                                                                       | Link                                                           |
| ----------------------------------------------------------------------------------------------- | -------------------------------------------------------------- |
| PaSST: Efficient Training of Audio Transformers with Patchout                                   | https://github.com/fschmid56/PaSST                             |
| Effective Pre-Training of Audio Transformers for Sound Event Detection                          | https://github.com/fschmid56/PretrainedSED                     |
| Florian Schmid (Audio Works)                                                                    | https://github.com/fschmid56?tab=repositories                  |
| Institute of Computational Perception                                                           | https://github.com/CPJKU                                       |
| Improving Audio Spectrogram Transformers for Sound Event Detection through Multi-Stage Training | https://arxiv.org/html/2408.00791v1                            |
| Audio Spectrogram Transformer (fine-tuned on AudioSet)                                          | https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593 |
| FAST: ==Fast Audio Spectrogram Transformer== \| ICASSP 2025                                     | https://youtu.be/U4h-8lBCpYA                                   |
| Harmonai Hangouts: LAION-CLAP with Yusong Wu & Ke Chen                                          | https://github.com/LAION-AI/CLAP                               |
| SonicMaster: Towards Controllable All-in-One Music Restoration and Mastering                    | https://arxiv.org/abs/2508.03448                               |

| Jupyter Notebook              | Link                                                                                                                   |
| ----------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| Transfer Learning with YAMNet | https://colab.research.google.com/gist/Vishesht27/df4ecd6aa06a31e94809201ada7780fb/transfer_learning_with_yamnet.ipynb |

| Data                          | Link                                              |
| ----------------------------- | ------------------------------------------------- |
| A lot of long-form video data | https://archive.org/details/movies?tab=collection |

YAMNet is a solid starting point considering it is pretrained on AudioSet
- can use it as a ==feature extractor== and train a classifier on top for artifact detection
- ideal for lightweight and fast, ideal for prototyping

BUT they struggle with ==long-range temporal dependencies== or ==contextual relationships== across time.

Transformers (like AST, PaSST, BEATs, or HTS-AT) are state of the art for audio classification:
- they model ==global context== across time and frequency
- they often outperform CNNs on AudioSet and other HEAR benchmarks
- they're more flexible for finetuning on custom-tasks

---
Way To Go
---
- **Start with [[YAMNet]]** - fast iteration, baseline results
- **Use YAMNet embeddings** to train a binary classifier (artifact vs clean)
- Once you have a working pipeline, **switch to AST, PaSST or BEATs** for fine-tuning
- **Evaluate both** - compare performance, latency and generalization


Learnt from ESC50
created ESC50 artifact with clean , 6 artifact classes
modified YAMNet model classifier for 7 classes and cross entropy loss for single label classifications
tried training only on esc50 for out of box reference

current step: using pretrained tensorflow AudioSet YAMNet weights for pytorch and perform classification for esc50artifact


| Models to deeply understand   |     |
| ----------------------------- | --- |
| YAMNet                        |     |
| MobileNetV1                   |     |
| MobileNetV3                   |     |
| Audio Spectrogram Transformer |     |
| FAST                          |     |
| Convolution Free Transformer  |     |
| PANN                          |     |
| PaSST                         |     |

we can play with 
hyperparameters, 
regularization - l1,l2,dropout, early stopping
slight variations in chunks before proceeding
architectural changes
dataset quality improvement
dataset is balanced class wise but dataset quality balancing (by listening)
contrastive learning (clean vs artifact)