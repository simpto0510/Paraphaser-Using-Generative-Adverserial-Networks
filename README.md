# Paraphaser Using Generative Adversarial Network


A full, end-to-end PyTorch implementation of DivGAN — a latent-conditioned Generative Adversarial Network for diverse paraphrase generation.
The repo reproduces the key ideas of “DivGAN: Towards Diverse Paraphrase Generation via Diversified Generative Adversarial Network” (Cao & Wan, 2020) and ships:

## ⚙️ Components
- CSV → HDF5/JSON pre-processing pipeline (Quora Question Pairs)	
- QQPDataset + DataLoaders (train / val split)	
### Generator
- Bi-GRU encoder + latent-conditioned GRU decoder
- Sequence-level representation for diversity loss	
### Discriminator
- CNN text encoder 
### Losses
- Adversarial (with label smoothing)
- Hinge-style diversity
- Training loop with gradient-clipping, FP16 toggle
- Evaluation metrics: corpus-level BLEU-4	✅
- Ready-to-run Kaggle notebook cell blocks	✅

## 🛠 Key dependencies
- PyTorch ≥ 2.0
- h5py, pandas, tqdm
- sacrebleu (evaluation)
- (BERTScore optional; commented out by default to keep memory low)

## References
- https://github.com/dev-chauhan/PQG-pytorch/tree/master

