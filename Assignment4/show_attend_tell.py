"""
Show, Attend and Tell: Neural Image Caption Generation with Visual Attention
Xu et al., 2015 — Soft Attention Implementation in PyTorch

Paper: https://arxiv.org/abs/1502.03044

Architecture Overview:
    1. Encoder: Pretrained ResNet extracts spatial feature maps (14x14xD)
    2. Attention: MLP computes attention weights over spatial locations
    3. Decoder: LSTM generates captions word-by-word using context vectors

Author note: Only soft (deterministic) attention is implemented here.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
import random
import os

# ─────────────────────────────────────────────────────────────
# Device
# ─────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# ─────────────────────────────────────────────────────────────
# 1. ENCODER
#    Uses ResNet-18 and strips the final avgpool + fc layers.
#    Output: (batch, num_pixels, encoder_dim)
#    where num_pixels = 14*14 = 196 for a 224x224 input.
# ─────────────────────────────────────────────────────────────
class Encoder(nn.Module):
    """
    Encodes an input image into a grid of feature vectors.

    ResNet-18 produces feature maps of shape (B, 512, 14, 14)
    for a 224x224 input after removing the last avgpool and fc layers.
    We reshape this to (B, 196, 512) — 196 spatial locations each
    described by a 512-dimensional vector.
    """

    def __init__(self, encoded_image_size: int = 14):
        super().__init__()
        self.enc_image_size = encoded_image_size

        # Load pretrained ResNet-18 (set weights=None to skip download in offline envs)
        try:
            resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        except Exception:
            print("  [Note] Could not download pretrained weights; using random init.")
            resnet = models.resnet18(weights=None)

        # Remove the adaptive average pool and the final fc layer.
        # We want spatial feature maps, not a single global vector.
        modules = list(resnet.children())[:-2]  # keep up to layer4
        self.resnet = nn.Sequential(*modules)

        # Adaptive pooling lets us control the output spatial size.
        self.adaptive_pool = nn.AdaptiveAvgPool2d(
            (encoded_image_size, encoded_image_size)
        )

        # We fine-tune the encoder during training (optional but helpful)
        self.fine_tune(fine_tune=True)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        images: (batch, 3, H, W)
        returns: (batch, num_pixels, encoder_dim)
                 e.g. (B, 196, 512)
        """
        features = self.resnet(images)            # (B, 512, H', W')
        features = self.adaptive_pool(features)   # (B, 512, 14, 14)

        B, C, H, W = features.shape
        # Reshape so each pixel is a row vector
        features = features.permute(0, 2, 3, 1)  # (B, 14, 14, 512)
        features = features.view(B, -1, C)        # (B, 196, 512)
        return features

    def fine_tune(self, fine_tune: bool = True):
        """Allow/prevent gradients in encoder layers."""
        # Freeze first two blocks, fine-tune the rest
        for p in self.resnet.parameters():
            p.requires_grad = False
        for child in list(self.resnet.children())[5:]:   # layer3, layer4
            for p in child.parameters():
                p.requires_grad = fine_tune


# ─────────────────────────────────────────────────────────────
# 2. ATTENTION MODULE (Soft Attention)
#    Computes attention weights α over the num_pixels locations.
#
#    e_t,i = W_a · tanh(W_enc · a_i + W_dec · h_{t-1})
#    α_t   = softmax(e_t)
#    ẑ_t   = Σ_i  α_{t,i} · a_i          (context vector)
# ─────────────────────────────────────────────────────────────
class Attention(nn.Module):
    """
    Soft (deterministic) attention over spatial encoder features.

    At each decoding step t, given:
        - encoder_out: all spatial feature vectors  (B, num_pixels, enc_dim)
        - decoder_hidden: LSTM hidden state at t-1  (B, dec_dim)

    Produces:
        - context: weighted sum of features          (B, enc_dim)
        - alpha:   attention weights                 (B, num_pixels)
    """

    def __init__(self, encoder_dim: int, decoder_dim: int, attention_dim: int):
        """
        encoder_dim:   dimension of encoder feature vectors (e.g. 512)
        decoder_dim:   dimension of LSTM hidden state       (e.g. 512)
        attention_dim: dimension of the attention MLP       (e.g. 256)
        """
        super().__init__()
        # Linear projections for encoder features and decoder hidden state
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)   # W_enc
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)   # W_dec
        # Final energy projection to scalar
        self.full_att = nn.Linear(attention_dim, 1)                 # W_a

    def forward(
        self,
        encoder_out: torch.Tensor,   # (B, num_pixels, enc_dim)
        decoder_hidden: torch.Tensor # (B, dec_dim)
    ):
        # Project encoder features: (B, num_pixels, att_dim)
        att1 = self.encoder_att(encoder_out)
        # Project decoder hidden state: (B, att_dim) → (B, 1, att_dim)
        att2 = self.decoder_att(decoder_hidden).unsqueeze(1)

        # Combine and compute energy scores: (B, num_pixels, 1) → (B, num_pixels)
        energy = self.full_att(torch.tanh(att1 + att2)).squeeze(2)

        # Softmax to get attention weights α (sum to 1 over pixels)
        alpha = F.softmax(energy, dim=1)  # (B, num_pixels)

        # Weighted sum of encoder features = context vector
        context = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (B, enc_dim)

        return context, alpha


# ─────────────────────────────────────────────────────────────
# 3. DECODER (LSTM with Attention)
#
#    At each step t:
#      1. Attend over image features → context vector ẑ_t
#      2. LSTM input = [embed(w_{t-1}) ; ẑ_t]
#      3. Predict next word from LSTM output
# ─────────────────────────────────────────────────────────────
class DecoderWithAttention(nn.Module):
    """
    LSTM decoder that generates captions word-by-word.
    At each step the decoder "looks" at the image via soft attention.
    """

    def __init__(
        self,
        attention_dim: int,
        embed_dim: int,
        decoder_dim: int,
        vocab_size: int,
        encoder_dim: int = 512,
        dropout: float = 0.5,
    ):
        """
        attention_dim: hidden size of attention MLP
        embed_dim:     word embedding dimension
        decoder_dim:   LSTM hidden state dimension
        vocab_size:    total vocabulary size
        encoder_dim:   dimension of CNN feature vectors
        dropout:       dropout rate on decoder output
        """
        super().__init__()

        self.encoder_dim  = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim    = embed_dim
        self.decoder_dim  = decoder_dim
        self.vocab_size   = vocab_size

        # Attention module
        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)

        # Word embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # LSTM cell (we use LSTMCell for step-by-step decoding)
        # Input = word embedding + context vector
        self.decode_step = nn.LSTMCell(
            input_size=embed_dim + encoder_dim,
            hidden_size=decoder_dim,
            bias=True
        )

        # Initialize LSTM state from mean-pooled encoder output
        self.init_h = nn.Linear(encoder_dim, decoder_dim)  # h_0 = f(mean_enc)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)  # c_0 = f(mean_enc)

        # "Doubly stochastic" gating (Section 4.2.1 of paper)
        # β_t gates how much the context contributes
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)

        # Output projection: hidden state → vocabulary logits
        self.fc = nn.Linear(decoder_dim, vocab_size)

        self.dropout = nn.Dropout(p=dropout)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Simple uniform initialization for embedding and fc layers."""
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def init_hidden_state(self, encoder_out: torch.Tensor):
        """
        Initialize LSTM h and c from mean of encoder features.
        encoder_out: (B, num_pixels, enc_dim)
        Returns h, c each of shape (B, decoder_dim)
        """
        mean_enc = encoder_out.mean(dim=1)   # (B, enc_dim)
        h = torch.tanh(self.init_h(mean_enc))
        c = torch.tanh(self.init_c(mean_enc))
        return h, c

    def forward(
        self,
        encoder_out: torch.Tensor,  # (B, num_pixels, enc_dim)
        captions: torch.Tensor,     # (B, max_len)  — token indices
        caption_lengths: torch.Tensor  # (B,) — true lengths
    ):
        """
        Teacher-forced forward pass.

        Returns:
            predictions:  (B, max_len-1, vocab_size)  — logits per step
            alphas:       (B, max_len-1, num_pixels)  — attention weights
        """
        B = encoder_out.size(0)
        max_len = captions.size(1) - 1   # we don't predict after <end>

        # Embed all caption tokens at once
        embeddings = self.embedding(captions)  # (B, max_len, embed_dim)

        # Initialize LSTM states
        h, c = self.init_hidden_state(encoder_out)

        predictions = torch.zeros(B, max_len, self.vocab_size).to(DEVICE)
        alphas      = torch.zeros(B, max_len, encoder_out.size(1)).to(DEVICE)

        for t in range(max_len):
            # ── Attention ──────────────────────────────────────────
            context, alpha = self.attention(encoder_out, h)
            # (B, enc_dim), (B, num_pixels)

            # Sigmoid gate on context (doubly stochastic attention)
            gate = torch.sigmoid(self.f_beta(h))   # (B, enc_dim)
            context = gate * context

            # ── LSTM step ──────────────────────────────────────────
            # Input = previous word embedding  ||  context vector
            lstm_input = torch.cat(
                [embeddings[:, t, :], context], dim=1
            )  # (B, embed_dim + enc_dim)

            h, c = self.decode_step(lstm_input, (h, c))  # each (B, dec_dim)

            # ── Predict ────────────────────────────────────────────
            preds = self.fc(self.dropout(h))   # (B, vocab_size)

            predictions[:, t, :] = preds
            alphas[:, t, :]      = alpha

        return predictions, alphas

    def sample(
        self,
        encoder_out: torch.Tensor,
        word2idx: dict,
        idx2word: dict,
        max_len: int = 20
    ):
        """
        Greedy decoding for inference (no teacher forcing).

        Returns:
            words:  list of predicted words
            alphas: (1, steps, num_pixels) attention maps
        """
        B = encoder_out.size(0)
        assert B == 1, "sample() expects batch size = 1"

        h, c = self.init_hidden_state(encoder_out)
        alphas = []

        # Start with <start> token
        word = torch.tensor([word2idx["<start>"]]).to(DEVICE)  # (1,)
        words = []

        for _ in range(max_len):
            emb = self.embedding(word)  # (1, embed_dim)

            context, alpha = self.attention(encoder_out, h)
            alphas.append(alpha)

            gate = torch.sigmoid(self.f_beta(h))
            context = gate * context

            lstm_input = torch.cat([emb, context], dim=1)
            h, c = self.decode_step(lstm_input, (h, c))

            logits = self.fc(h)             # (1, vocab_size)
            pred   = logits.argmax(dim=1)   # (1,)

            token = idx2word[pred.item()]
            if token == "<end>":
                break
            words.append(token)
            word = pred

        alphas = torch.stack(alphas, dim=1)  # (1, steps, num_pixels)
        return words, alphas


# ─────────────────────────────────────────────────────────────
# 4. VOCABULARY HELPER
# ─────────────────────────────────────────────────────────────
class Vocabulary:
    """Simple vocabulary that maps words ↔ indices."""

    PAD = "<pad>"
    START = "<start>"
    END = "<end>"
    UNK = "<unk>"

    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self._add(self.PAD)    # 0
        self._add(self.START)  # 1
        self._add(self.END)    # 2
        self._add(self.UNK)    # 3

    def _add(self, word: str):
        if word not in self.word2idx:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word

    def build_from_captions(self, captions: list):
        for cap in captions:
            for word in cap.lower().split():
                self._add(word)

    def encode(self, caption: str, max_len: int):
        """Convert caption string to padded index list."""
        tokens = (
            [self.word2idx[self.START]]
            + [self.word2idx.get(w, self.word2idx[self.UNK])
               for w in caption.lower().split()]
            + [self.word2idx[self.END]]
        )
        # Pad or truncate to max_len
        if len(tokens) < max_len:
            tokens += [self.word2idx[self.PAD]] * (max_len - len(tokens))
        else:
            tokens = tokens[:max_len]
        return tokens

    def __len__(self):
        return len(self.word2idx)


# ─────────────────────────────────────────────────────────────
# 5. DUMMY DATASET
#    Generates synthetic (image, caption) pairs so the code
#    runs without needing to download MS-COCO.
# ─────────────────────────────────────────────────────────────
DUMMY_CAPTIONS = [
    "a dog playing in the park",
    "a cat sitting on a table",
    "a person riding a bicycle",
    "a red car on a highway",
    "two birds flying in the sky",
    "a child eating an ice cream",
    "a woman reading a book",
    "a large ship on the ocean",
    "a man running on the beach",
    "colorful flowers in a garden",
]

class DummyDataset(Dataset):
    """
    Creates random RGB images paired with simple captions.
    In a real project you'd replace this with MS-COCO or Flickr30k.
    """

    def __init__(self, vocab: Vocabulary, num_samples: int = 200, max_len: int = 20):
        self.vocab      = vocab
        self.num_samples = num_samples
        self.max_len    = max_len

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],   # ImageNet stats
                std=[0.229, 0.224, 0.225]
            ),
        ])

        # Assign each sample a random caption
        self.captions = [
            random.choice(DUMMY_CAPTIONS) for _ in range(num_samples)
        ]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Random image (uniform noise) as a PIL image
        arr = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(arr, mode="RGB")
        img = self.transform(img)   # (3, 224, 224)

        # Encode caption
        cap = self.captions[idx]
        encoded = self.vocab.encode(cap, self.max_len)
        caption_tensor = torch.tensor(encoded, dtype=torch.long)

        # True length (capped at max_len)
        length = min(
            len(cap.lower().split()) + 2,  # +2 for <start> and <end>
            self.max_len
        )

        return img, caption_tensor, torch.tensor(length)


# ─────────────────────────────────────────────────────────────
# 6. TRAINING LOOP
# ─────────────────────────────────────────────────────────────
def train_one_epoch(encoder, decoder, loader, optimizer, criterion):
    """Run one full pass over the training data."""
    encoder.train()
    decoder.train()
    total_loss = 0.0

    for batch_idx, (images, captions, lengths) in enumerate(loader):
        images   = images.to(DEVICE)
        captions = captions.to(DEVICE)
        lengths  = lengths.to(DEVICE)

        # ── Forward pass ────────────────────────────────────
        features = encoder(images)                            # (B, 196, 512)
        predictions, alphas = decoder(features, captions, lengths)
        # predictions: (B, max_len-1, vocab_size)

        # ── Loss ─────────────────────────────────────────────
        # Target: captions shifted by 1 (we predict the NEXT token)
        targets = captions[:, 1:]   # (B, max_len-1)

        # Flatten for CrossEntropy: (B*(max_len-1), vocab_size)
        loss = criterion(
            predictions.reshape(-1, decoder.vocab_size),
            targets.reshape(-1)
        )

        # Doubly-stochastic attention regularization (Eq. 14 in paper)
        # Encourages attention weights to sum to ~1 over time for each pixel
        loss += 1.0 * ((1.0 - alphas.sum(dim=1)) ** 2).mean()

        # ── Backward pass ────────────────────────────────────
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping to prevent exploding gradients
        nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=5.0)
        nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=5.0)

        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def train(epochs: int = 5):
    """Full training pipeline."""

    # ── Build vocabulary ──────────────────────────────────────
    vocab = Vocabulary()
    vocab.build_from_captions(DUMMY_CAPTIONS)
    print(f"Vocabulary size: {len(vocab)}")

    # ── Dataset & DataLoader ──────────────────────────────────
    dataset = DummyDataset(vocab, num_samples=256, max_len=20)
    loader  = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)

    # ── Model ─────────────────────────────────────────────────
    encoder = Encoder(encoded_image_size=14).to(DEVICE)
    decoder = DecoderWithAttention(
        attention_dim=256,
        embed_dim=256,
        decoder_dim=512,
        vocab_size=len(vocab),
        encoder_dim=512,
        dropout=0.5,
    ).to(DEVICE)

    # ── Optimiser ─────────────────────────────────────────────
    # Fine-tuning encoder at a lower learning rate
    params = [
        {"params": decoder.parameters()},
        {"params": filter(lambda p: p.requires_grad, encoder.parameters()),
         "lr": 1e-4}
    ]
    optimizer = torch.optim.Adam(params, lr=4e-4)

    # Ignore padding index in loss
    criterion = nn.CrossEntropyLoss(
        ignore_index=vocab.word2idx[Vocabulary.PAD]
    ).to(DEVICE)

    # ── Training loop ─────────────────────────────────────────
    print("\nStarting training …")
    for epoch in range(1, epochs + 1):
        loss = train_one_epoch(encoder, decoder, loader, optimizer, criterion)
        print(f"  Epoch {epoch}/{epochs}  |  Loss: {loss:.4f}")

    print("\nTraining complete.")
    return encoder, decoder, vocab


# ─────────────────────────────────────────────────────────────
# 7. INFERENCE — GREEDY DECODING
# ─────────────────────────────────────────────────────────────
def generate_caption(encoder, decoder, image_tensor, vocab, max_len=20):
    """
    Generate a caption for a single image tensor.

    image_tensor: (3, 224, 224) — already normalized
    Returns: caption string, attention maps (1, steps, num_pixels)
    """
    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        image   = image_tensor.unsqueeze(0).to(DEVICE)   # (1, 3, 224, 224)
        features = encoder(image)                         # (1, 196, 512)
        words, alphas = decoder.sample(
            features, vocab.word2idx, vocab.idx2word, max_len=max_len
        )

    caption = " ".join(words)
    return caption, alphas


# ─────────────────────────────────────────────────────────────
# 8. ATTENTION VISUALISATION
# ─────────────────────────────────────────────────────────────
def visualize_attention(image_tensor, caption_words, alphas, save_path="attention_vis.png"):
    """
    Plot the image alongside attention maps for each generated word.

    image_tensor: (3, 224, 224) tensor (normalised)
    caption_words: list of predicted word strings
    alphas: (1, steps, num_pixels)  e.g. num_pixels=196 → 14x14
    """
    # De-normalise image for display
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img  = (image_tensor * std + mean).clamp(0, 1)
    img  = img.permute(1, 2, 0).numpy()   # (H, W, 3)

    alphas = alphas.squeeze(0).cpu().numpy()   # (steps, num_pixels)
    num_pixels = alphas.shape[1]
    grid_size  = int(num_pixels ** 0.5)        # 14 for 196 pixels

    n_words = len(caption_words)
    cols    = min(n_words, 5)
    rows    = (n_words + cols - 1) // cols + 1   # +1 for original image row

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = np.array(axes).reshape(-1)

    # Show original image first
    axes[0].imshow(img)
    axes[0].set_title("Input Image", fontsize=9)
    axes[0].axis("off")

    for i, word in enumerate(caption_words):
        ax  = axes[i + 1]
        att = alphas[i].reshape(grid_size, grid_size)  # (14, 14)

        # Upsample attention map to image size
        att_up = np.array(
            Image.fromarray(att).resize((224, 224), resample=Image.BILINEAR)
        )
        ax.imshow(img)
        ax.imshow(att_up, alpha=0.5, cmap=cm.hot)
        ax.set_title(word, fontsize=9)
        ax.axis("off")

    # Hide unused axes
    for j in range(n_words + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Attention visualization saved → {save_path}")


# ─────────────────────────────────────────────────────────────
# 9. MAIN
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # ── Train ─────────────────────────────────────────────────
    encoder, decoder, vocab = train(epochs=5)

    # ── Inference on a random test image ──────────────────────
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    # Random test image (replace with a real image path if you have one)
    arr       = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    test_img  = transform(Image.fromarray(arr, "RGB"))

    caption, alphas = generate_caption(encoder, decoder, test_img, vocab)
    print(f"\nGenerated caption: \"{caption}\"")

    # ── Attention visualisation ────────────────────────────────
    words = caption.split()
    if words:
        visualize_attention(test_img, words, alphas, save_path="attention_vis.png")
    else:
        print("(Empty caption — skipping visualisation.)")
