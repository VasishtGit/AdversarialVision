# Model Card: ResNet-18 (CIFAR-100)

**Generated Date:** 2026-10-19 12:10:46  
**Framework:** AdversarialVision  
**License:** MIT  

---

## 1. Model Details
- **Architecture:** ResNet-18 (CIFAR-100)
- **Dataset:** CIFAR-100 (45k train, 5k val, 10k test)
- **Input Space:** Strictly $[0.0, 1.0]$ physical pixel bounds with internal differentiable `NormalizeLayer`
- **Training Strategy:** `TRADES (beta=6.0)`

---

## 2. Intended Use & Threat Models
- **Clean Vision Tasks:** CIFAR-100 100-class fine-grained classification
- **Adversarial Threat Model:** $L_\infty$ bounded perturbations (standard budget $\epsilon = 8/255$)
- **Corruptions Scope:** Common natural corruptions across Noise, Blur, Compression, Lighting, and Geometry

---

## 3. Empirical Performance Summary
- **Clean Test Top-1 Accuracy:** 77.84%
- **Clean Test Top-5 Accuracy:** N/A
- **Test Cross-Entropy Loss:** 0.9134
- **Inference Latency:** 1.278 ms / sample

---

## 4. Defense Strategies Available
1. **Adversarial Training:** Min-Max PGD-AT (Madry et al.) & TRADES (Zhang et al.)
2. **Inference-Time Preprocessing:** Bilateral, Median ($k=3$), JPEG Quantization ($Q=75$)
3. **Certified Defense:** Randomized Smoothing with Clopper-Pearson certified $L_2$ radius
4. **Adversarial Detection:** Energy-based out-of-distribution / perturbation scoring

---

## 5. Caveats & Ethical Considerations
- Deep neural network robustness is empirical and subject to adaptive white-box attacks.
- Robust accuracy claims should be validated with multi-attack suites including AutoAttack.
