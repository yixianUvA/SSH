# SSH: Sparse Spectrum Adaptation via Discrete Hartley Transformation

**Official implementation of the SSH method from the paper:**

> **SSH: Sparse Spectrum Adaptation via Discrete Hartley Transformation**  
> *Yixian Shen, Qi Bi, Jia-Hong Huang, Hongyi Zhu, Andy D. Pimentel, Anuj Pathania*  
> University of Amsterdam  
> [arXiv:2502.05539](https://arxiv.org/abs/2502.05539)

---

## 📌 Overview

SSH (Sparse Spectrum Adaptation via Discrete Hartley Transformation) is a novel parameter-efficient fine-tuning (PEFT) method for large foundation models (LLMs, ViTs, etc.). It works by transforming model weights into the Hartley spectral domain, selecting only the most informative frequency components based on energy, and then applying updates through an inverse DHT. SSH outperforms prior PEFT methods like LoRA and FourierFT in terms of both performance and computational efficiency.

---

## ✨ Highlights

- **Parameter Efficiency**: Achieves strong performance with 10–100× fewer parameters than LoRA.
- **Hartley Transform**: Avoids complex numbers and enables symmetric forward/inverse transforms.
- **Energy-Based Frequency Selection**: Updates only the most informative spectral components.
- **Applicable Across Domains**: Strong results on NLP, vision, and multi-modal tasks.

---

## 📈 Performance

SSH achieves:

- **GLUE Benchmark (RoBERTa-Large)**: Best average performance with only 0.036M parameters.
- **Instruction Tuning (LLaMA3.1-8B)**: GPT-4 score of 7.71 with 0.055M params (vs. 183.3M for LoRA).
- **Image Classification (ViT-L)**: Matches full finetuning on EuroSAT and OxfordPets with 20–30× fewer parameters.
- **Text Summarization (BART-Large)**: ROUGE-L of 42.13 on CNN/DailyMail with just 0.21M params.

For detailed tables and figures, see the [paper](https://arxiv.org/abs/2502.05539).

---

## 🧠 Method

SSH consists of three main steps:
1. Apply **2D Discrete Hartley Transform (DHT)** to pretrained weights.
2. Select top-δ% frequencies by energy and randomly sample the rest.
3. Learn only selected spectral coefficients; project back to the spatial domain using inverse DHT.

```python
# Pseudocode (simplified)
W_f = DHT(W_base)
H = select_frequencies(W_f, energy_ratio=δ)
∆W = iDHT(H) * α
W_new = W_base + ∆W
```

---

## 📦 Installation

```bash
git clone https://github.com/yixianUvA/SSH.git
cd SSH
pip install -r requirements.txt
```

> Requires Python 3.8+ and PyTorch. May also use `transformers`, `datasets`, and `scipy`.

---

## 🚀 Usage

SSH can be applied to any linear layer of a pretrained transformer or vision model.

```python
from ssh import SSHLayer

ssh_layer = SSHLayer(
    base_weight=W_base,
    num_trainable=750,
    energy_ratio=0.7,
    alpha=1.0
)
output = ssh_layer(input)
```

More examples and scripts for GLUE, GPT-2, ViT, and LLaMA are in the `examples/` directory.

---

## 📊 Results

SSH has been tested across:

- **NLP**: GLUE (RoBERTa), E2E NLG (GPT-2), summarization (BART)
- **Vision**: CIFAR-100, DTD, EuroSAT, OxfordPets (ViT)
- **Multi-modal**: Instruction tuning with LLaMA 7B/13B/8B on Alpaca
- **Reasoning**: GSM8K on LLaMA3.1-8B

Check the `results/` folder or [paper appendix](https://arxiv.org/abs/2502.05539) for all benchmark results.

---

## 📄 Citation

If you use SSH in your research, please cite:

```bibtex
@article{shen2025ssh,
  title={SSH: Sparse Spectrum Adaptation via Discrete Hartley Transformation},
  author={Shen, Yixian and Bi, Qi and Huang, Jia-Hong and Zhu, Hongyi and Pimentel, Andy D. and Pathania, Anuj},
  journal={arXiv preprint arXiv:2502.05539},
  year={2025}
}
```

---

## 📩 Contact

For questions or collaborations, please contact [y.shen@uva.nl](mailto:y.shen@uva.nl) or open an issue.

---

## 📝 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
