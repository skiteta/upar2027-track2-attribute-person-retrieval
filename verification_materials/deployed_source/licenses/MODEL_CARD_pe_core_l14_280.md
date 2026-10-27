---
tags:
- image-feature-extraction
- timm
- transformers
library_name: timm
license: apache-2.0
---
# Model Details

This is a `timm` remapped, image encoder only variant of the original weights.

[\[📃 Tech Report\]](https://arxiv.org/abs/2504.13181)
[\[📂 Github\]](https://github.com/facebookresearch/perception_models/)

Perception Encoder (PE) is a state-of-the-art encoder for image and video understanding trained via simple vision-language learning. It was introduced in "[Perception Encoder: The best visual embeddings
are not at the output of the network](https://ai.meta.com/research/publications/perception-encoder-the-best-visual-embeddings-are-not-at-the-output-of-the-network/)".

**Model Developer**: Meta

**Model Overview**: Perception Encoder (PE) is a family of large-scale vision encoder models with state-of-the-art performance on a large variety of vision tasks. By using a robust contrastive pretraining recipe and finetuning on synthetically aligned videos, PE not only outperforms all existing models on classification and retrieval, but it also internally produces strong, general features that scale for downstream tasks. PE unlocks the ability for large-scale contrastive pretraining to transfer to downstream tasks with alignment tuning to capitalize on those general features.

<img src="https://huggingface.co/facebook/PE-Core-G14-448/resolve/main/docs/pe_image1.png" style="width: 100%; margin: 0 auto; display: block;" />

## Perception Encoder: Spatial
PE spatial similarly takes the strong spatial performance from the intermediate layers of PE core and aligns it to the end using a simple frozen teacher self-distillation loss and further refines with a novel SAM 2.1 mask-based learning strategy. PE spatial performs well on dense prediction tasks such as detection.

And despite being a short finetuning step using PE core's intermediate layers as a teacher (a pure CLIP model with a global loss) plus a little bit of refinement with SAM, the resulting feature space is quite detailed and well-aligned. Here we picture the PCA of the last layer features mapped to LCh color space (see the paper for more details):

PE spatial also has nuanced semantic correspondences between objects thanks to its CLIP pretraining. Here we show again PCA but only for the tokens not masked. PE spatial shows correspondence between parts like the first image cats' heads, backs, and legs. Additionally, PE spatial can show more nuanced correspondences like for the last two images, where the red/blue directions still denote parts, but the lightness/darkness directions now indicate semantics (i.e., dog/cat breed):


We release one checkpoint for PE spatial so far:
| Encoder | Checkpoint | ADE20k <br/> Linear Probe <br/> 448px w/o TTA | LVIS <br /> Mask R-CNN 1024px <br /> Box / Mask mAP | COCO <br/> DETA 1728px <br /> Box mAP |
|:---:|:---:|:---:|:---:|:---:|
| **G/14** 448px | [PE-Spatial-G14-448](https://huggingface.co/facebook/PE-Spatial-G14-448) | 49.3 | 54.2 / 49.3 | 65.5

See paper for full set of evaluations and fair comparison to other works.

# Citation
If you find our code useful for your research, please consider citing:
    
```
@article{bolya2025PerceptionEncoder,
  title={Perception Encoder: The best visual embeddings are not at the output of the network},
  author={Daniel Bolya and Po-Yao Huang and Peize Sun and Jang Hyun Cho and Andrea Madotto and Chen Wei and Tengyu Ma and Jiale Zhi and Jathushan Rajasegaran and Hanoona Rasheed and Junke Wang and Marco Monteiro and Hu Xu and Shiyu Dong and Nikhila Ravi and Daniel Li and Piotr Doll{\'a}r and Christoph Feichtenhofer},
  journal={arXiv},
  year={2025}
}

@article{cho2025PerceptionLM,
  title={PerceptionLM: Open-Access Data and Models for Detailed Visual Understanding},
  author={Jang Hyun Cho and Andrea Madotto and Effrosyni Mavroudi and Triantafyllos Afouras and Tushar Nagarajan and Muhammad Maaz and Yale Song and Tengyu Ma and Shuming Hu and Hanoona Rasheed and Peize Sun and Po-Yao Huang and Daniel Bolya and Suyog Jain and Miguel Martin and Huiyu Wang and Nikhila Ravi and Shashank Jain and Temmy Stark and Shane Moon and Babak Damavandi and Vivian Lee and Andrew Westbury and Salman Khan and Philipp Kr\"{a}henb\"{u}hl and Piotr Doll{\'a}r and Lorenzo Torresani and Kristen Grauman and Christoph Feichtenhofer},
  journal={arXiv},
  year={2025}
}
```
