# 🌊 Diffusion Factor Model with ESG Constrains

<p align="center">
<img src="assets/esg_portfolio_performance.png" alt="ESG Portfolio Performance Demo" width="700"/>
</p>

This repository extends the Diffusion Factor Model to construct and evaluate high-dimensional, ESG-constrained investment portfolios.

## 📝 Summary

This project applies a novel Diffusion Factor Model (DFM) to the critical challenge of sustainable investing. By adapting diffusion models to generate realistic financial returns with latent factor structure, we aim to significantly improve the estimation of expected returns and covariance matrices. These improved estimates are then used to construct mean-variance optimal portfolios subject to Environmental, Social, and Governance (ESG) constraints. Our work demonstrates that this combination of generative AI and sustainable finance principles can lead to the construction of more efficient and robust "green" portfolios, achieving superior out-of-sample performance.

## ✨ Features

- 📊 Diffusion models adapted for financial data with factor structure
- 🌿 ESG-constrained portfolio optimization
- 🔄 Rolling window backtesting engine for out-of-sample evaluation
- 🧼 Advanced data preprocessing (merging, winsorization)
- 📈 Full suite of portfolio performance metrics (Sharpe Ratio, Max Drawdown, etc.)

## 🔧 Installation

```bash
git clone https://github.com/ZihaoYan7/diffusion_factor_model_with_esg_constrains.git
cd diffusion-factor-model-esg
pip install -r requirements.txt
```

For portfolio optimization, this project uses cvxpy. A high-performance solver like MOSEK is recommended and requires a license (free for academic use).

## 📁 Project Structure

```
diffusion_factor_model_with_esg_constrains/
├── assets/                      # Images for README (e.g., performance charts)
├── config/                      # Configuration settings for experiments
├── data_return_esg/             # All input data
│   ├── raw/                     # Raw return and ESG data (.csv files)
│   └── processed/               # Processed and winsorized data
├── diffusion_factor_model/      # Core DFM model implementation (from original repo)
├── eval/                        # Original evaluation modules
├── results/                     # All outputs from experiments (created automatically)
│   ├── trained_models/          # Trained DFM models for each backtest period
│   ├── generated_samples/       # Generated return samples for analysis
│   └── backtest_reports/        # Final performance charts and metrics
├── samples/                     # Generated samples from DFM (created automatically)
├── data_preprocessing.py        # (New) Script for merging, cleaning, and winsorizing data
├── esg_optimizer.py             # (New) ESG-constrained portfolio optimization module
├── backtester.py                # (New) Rolling window backtesting engine
├── main.py                      # (Modified) Main script to run ESG backtesting experiments
└── requirements.txt             # Project dependencies
```

## 🚀 Running Experiments

# Run the preprocessing script to merge, clean, and winsorize the raw data.

python data_preprocessing.py
```This will create a unified data file (e.g., `final_data.parquet`) in the `data/processed/` directory, which will be used for all subsequent steps.

# Adjust the parameters in your configuration file (e.g., `config/esg_experiment.yaml`) to define the backtest period and ESG constraint parameters. Then, run the main experiment script:

```bash
# Run the ESG portfolio backtesting experiment
python main.py --config_path /path/to/your/esg_experiment.yaml --gpu 0

The script will perform the rolling window backtest year by year. Trained models will be saved in model_results/, generated samples in samples/, and final performance reports will be generated at the end of the run.

## 📊 Evaluation

This repository evaluates the framework from two perspectives:
1. Model-Level Evaluation: Using modules from the eval/ directory to assess the quality of generated returns and the accuracy of latent factor recovery.
2. Portfolio-Level Evaluation: Using the new backtester.py module to assess the economic significance of the approach. This includes out-of-sample performance of ESG-constrained portfolios compared against various benchmarks.
<p align="center">
<img src="assets/esg_cumulative_returns.png">
</p>
<p align="center">
<em>Example: Cumulative log-returns of the DFM-based ESG portfolio vs. benchmarks.</em>
</p>

## 📚 Citation

```
@article{chen2025diffusion,
  title={Diffusion Factor Models: Generating High-Dimensional Returns with Factor Structure},
  author={Chen, Minshuo and Xu, Renyuan and Xu, Yumin and Zhang, Ruixun},
  journal={arXiv preprint arXiv:2504.06566},
  year={2025}
}

@article{lo2025performance,
  title={Performance Attribution for Portfolio Constraints},
  author={Lo, Andrew W and Zhang, Ruixun},
  journal={Management Science},
  year={2025}
}
```
