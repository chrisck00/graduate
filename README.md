# LLM Model Merging
## Model Merging with Adaptive Drop Ratebased on Weight Importance

<img width="1013" height="248" alt="Image" src="https://github.com/user-attachments/assets/bab8ded9-9e35-4030-83eb-9196e6af08b9" />

### Adaptive Drop Rate Merging

---

Large Language Models (LLMs) are fine-tuned on pre-trained models for task-specific performance, but managing multiple fine-tuned models is costly and integrating their capabilities is challenging. **'Adaptive Drop Rate (ADR) Merging'** addresses this by considering layer-wise importance: delta parameters with higher RMS are kept with lower drop rates, while less important layers are dropped more. Models are then merged using sign election and disjoint merging, followed by rescaling. Evaluations on NLU benchmarks with LoRA modules show that ADR Merging outperforms existing baselines such as TIES-Merging and DARE.

---


대규모 언어 모델(LLM)은 사전 학습 모델을 기반으로 특정 과제에 맞게 미세 조정되지만, 여러 미세 조정 모델을 관리하면 메모리와 시간 비용이 크고 모델 능력을 통합하기 어렵다. **'적응형 드롭율 병합(ADR Merging)'**은 레이어 단위 중요도를 고려하여 변화가 큰 레이어의 파라미터는 낮은 드롭율로 유지하고, 중요도가 낮은 레이어는 더 많이 드롭한다. 이후 부호 선정 및 서로소 병합 후 재스케일링하여 모델을 병합한다. LoRA 모듈을 활용한 NLU 벤치마크 평가에서 ADR Merging은 기존 TIES-Merging, DARE 등보다 우수한 성능을 보여준다.
