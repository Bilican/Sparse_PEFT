# Sparse_PEFT: Exploring Sparsity for Parameter-Efficient Fine-Tuning | [[Paper]](https://arxiv.org/abs/2505.12532)

<div align="center">
  <table>
    <tr>
      <td><img src="assets/dataset/colorful_sneaker.jpg" width="120"></td>
      <td><img src="assets/dataset/wolf_plushie.jpg" width="120"></td>
      <td><img src="assets/dataset/dog6.jpg" width="120"></td>
      <td><img src="assets/dataset/cat2.jpg" width="120"></td>
      <td><img src="assets/dataset/dog8.jpg" width="120"></td>
    </tr>
    <tr>
      <td><img src="assets/waveft/prompt_9598_sample_2_seed_44.png" width="120"></td>
      <td><img src="assets/waveft/prompt_5728_sample_1_seed_43.png" width="120"></td>
      <td><img src="assets/waveft/prompt_7595_sample_0_seed_42.png" width="120"></td>
      <td><img src="assets/waveft/prompt_4769_sample_3_seed_45.png" width="120"></td>
      <td><img src="assets/waveft/prompt_1569_sample_3_seed_45.png" width="120"></td>
    </tr>
    <tr>
      <td><img src="assets/lora/prompt_1258_sample_2_seed_44.png" width="120"></td>
      <td><img src="assets/lora/prompt_1757_sample_1_seed_43.png" width="120"></td>
      <td><img src="assets/lora/prompt_2257_sample_0_seed_42.png" width="120"></td>
      <td><img src="assets/lora/prompt_6648_sample_3_seed_45.png" width="120"></td>
      <td><img src="assets/lora/prompt_8189_sample_3_seed_45.png" width="120"></td>
    </tr>
  </table>
  <p>The original images (top), WaveFT results (middle), LoRA results (bottom).</p>
</div>

## Overview

This repository contains the implementation and results of our research on sparse parameter-efficient fine-tuning methods for large pre-trained models. We explore the benefits of incorporating structured sparsity into PEFT approaches to achieve better parameter efficiency while maintaining or improving performance.


Note: There are some issues that are going to be fixed shortly.

## Key Contributions

- Introduction of novel sparse PEFT methods that combine the benefits of sparse neural networks with parameter-efficient fine-tuning
- Comprehensive evaluation of existing and novel PEFT methods across multiple tasks and model architectures
- Development of WaveFT, a wavelet-based approach to sparse fine-tuning
- Extensive comparison with existing methods like LoRA (Low-Rank Adaptation)
- Analysis of performance vs. parameter count trade-offs

## Method

<div align="center">
  <img src="assets/method-1.png" width="700">
</div>

## Results

### PEFT Method Comparison

<div align="center">
  <img src="assets/peft_comparison.png" width="600">
</div>

Comparison of different PEFT methods including our novel sparse approaches against baseline methods.

### Rank vs Sparsity Trade-off

<div align="center">
  <img src="assets/rank_vs_p_combined_with_full_rank_linear.png" width="600">
</div>

Analysis of the relationship between rank and sparsity parameters in our methods, showing how different configurations affect model performance.

## WaveFT: Wavelet-based Fine-Tuning

<div align="center">
  <img src="assets/plot_waveft_scale_metrics.png" width="600">
</div>

WaveFT leverages wavelet transformations to identify important parameter subspaces for efficient fine-tuning, achieving strong results with minimal parameter updates.

## Performance Metrics

<div align="center">
  <img src="assets/performance_metrics_plot.png" width="600">
</div>

Comprehensive evaluation metrics across different tasks and model configurations.

## Directory Structure

- `assets/`: Visualizations and figures
  - `waveft/`: WaveFT-specific results and visualizations
  - `lora/`: LoRA-specific results and visualizations
  - `dataset/`: Dataset examples and visualizations

## Dataset Acknowledgment

The dataset used in this work is from the [DreamBooth repository](https://github.com/google/dreambooth) by Google. We use their dataset of subjects for our fine-tuning experiments to maintain consistency with prior work and enable fair comparison.

## Citation

If you find this work useful, please cite our paper:

```bibtex
@misc{bilican2025exploringsparsityparameterefficient,
      title={Exploring Sparsity for Parameter Efficient Fine Tuning Using Wavelets}, 
      author={Ahmet Bilican and M. Akın Yılmaz and A. Murat Tekalp and R. Gökberk Cinbiş},
      year={2025},
      eprint={2505.12532},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2505.12532}, 
}
```

## License

This project is licensed under the [Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/) license.



