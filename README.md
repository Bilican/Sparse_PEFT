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

## Method

<div align="center">
  <img src="assets/method-1.png" width="700">
</div>


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

- `assets/`: Visualizations and figures used in the paper
  - `waveft/`: WaveFT-specific results and visualizations
  - `lora/`: LoRA-specific results and visualizations
  - `dataset/`: Sample images used in the visualizations
- `dataset/`: Full dataset of subject images organized by category
- `peft/`: Modified version of Hugging Face PEFT to include WaveFT adapter. 
- `PyTorch-Wavelet-Toolbox-Custom/`: Modified version of the PyTorch Wavelet Toolbox for our implementation
- `sdxl_dreambooth/`: Code for fine-tuning Stable Diffusion XL with our methods
- `example_outputs/`: Example generation outputs and configuration files
- `requirements.txt`: Python dependencies needed to run the code

## Dataset Acknowledgment

The dataset used in this work is from the [DreamBooth repository](https://github.com/google/dreambooth) by Google. We use their dataset of subjects for our fine-tuning experiments to maintain consistency with prior work and enable fair comparison.

## Acknowledgements

This work utilizes the following open-source libraries:

- **Hugging Face PEFT**: A library for state-of-the-art parameter-efficient fine-tuning methods.
  - Repository: [https://github.com/huggingface/peft](https://github.com/huggingface/peft)
  - Citation:
    ```bibtex
    @Misc{peft,
      title =        {PEFT: State-of-the-art Parameter-Efficient Fine-Tuning methods},
      author =       {Sourab Mangrulkar and Sylvain Gugger and Lysandre Debut and Younes Belkada and Sayak Paul and Benjamin Bossan},
      howpublished = {\url{https://github.com/huggingface/peft}},
      year =         {2022}
    }
    ```

- **PyTorch Wavelet Toolbox (ptwt)**: A toolbox for differentiable fast wavelet transforms in PyTorch with GPU support.
  - Repository: [https://github.com/v0lta/PyTorch-Wavelet-Toolbox](https://github.com/v0lta/PyTorch-Wavelet-Toolbox)
  - Citation:
    ```bibtex
    @article{JMLR:v25:23-0636,
      author  = {Moritz Wolter and Felix Blanke and Jochen Garcke and Charles Tapley Hoyt},
      title   = {ptwt - The PyTorch Wavelet Toolbox},
      journal = {Journal of Machine Learning Research},
      year    = {2024},
      volume  = {25},
      number  = {80},
      pages   = {1--7},
      url     = {http://jmlr.org/papers/v25/23-0636.html}
    }
    ```

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



