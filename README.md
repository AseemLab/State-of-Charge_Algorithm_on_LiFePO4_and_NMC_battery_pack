# Advanced State of Charge Algorithm for LiFePO4 and NMC Battery Packs

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-latest-green.svg)](https://scikit-learn.org/)

A comprehensive machine learning approach for precise State of Charge (SoC) estimation in lithium-ion battery packs, specifically targeting LiFePO4 and NMC battery chemistries.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Battery Chemistries](#battery-chemistries)
- [Methodology](#methodology)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Models](#models)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Overview

This project implements an advanced State of Charge estimation algorithm using artificial neural networks combined with boosting algorithms. The system processes real Battery Management System (BMS) data to predict battery SoC with higher accuracy than conventional methods like Coulomb counting and voltage-based techniques.

### Key Objectives
- Develop machine learning models for precise SoC estimation
- Handle complex, nonlinear battery behavior patterns
- Account for temperature variations and charge/discharge cycles
- Compare multiple algorithmic approaches for optimal performance

##  Features

- **Multi-Algorithm Approach**: Combines ANN with XGBoost, AdaBoost, and Bayesian Ridge
- **Real BMS Data Processing**: Handles 30+ GB of high-frequency battery data
- **Temperature Compensation**: Accounts for thermal variations in battery performance
- **Robust Data Pipeline**: Advanced preprocessing and feature extraction
- **Performance Metrics**: Comprehensive evaluation using MSE and validation curves
- **Battery Chemistry Specific**: Optimized for both LiFePO4 and NMC batteries

##  Battery Chemistries

### LiFePO4 (Lithium Iron Phosphate)
- Flat voltage discharge curves
- Excellent thermal stability
- Long cycle life and safe operation
- Challenging for voltage-based SoC estimation

### NMC (Nickel Manganese Cobalt)
- Higher energy density
- Better high-temperature performance
- More sensitive OCV-SoC relationship
- Substantial capacity degradation over time

## Methodology

The project employs a hybrid approach combining:

1. **Artificial Neural Networks** (3-4 layer architectures)
2. **Boosting Algorithms**:
   - XGBoost (Extreme Gradient Boosting)
   - AdaBoost (Adaptive Boosting)
   - Bayesian Ridge Regression
3. **Advanced Data Processing**:
   - Microsecond-precision timestamping
   - Multi-parameter feature extraction
   - Data augmentation techniques

## Installation

### Prerequisites
```bash
Python 3.7+
TensorFlow 2.x
scikit-learn
XGBoost
pandas
numpy
matplotlib
```

### Setup
```bash
# Clone the repository
git clone https://github.com/AseemLab/soc-estimation-liion-batteries.git
cd soc-estimation-liion-batteries

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

##  Usage

### Basic Example
```python
from soc_estimation import SOCEstimator
from data_preprocessing import BMSDataProcessor

# Initialize data processor
processor = BMSDataProcessor()
processed_data = processor.load_and_clean('path/to/bms_data.csv')

# Initialize SoC estimator
estimator = SOCEstimator(model_type='xgboost')
estimator.train(processed_data)

# Make predictions
soc_predictions = estimator.predict(test_data)
```

### Training Custom Models
```python
# Train multiple models for comparison
models = ['xgboost', 'adaboost', 'bayesian_ridge']
results = {}

for model_type in models:
    estimator = SOCEstimator(model_type=model_type)
    estimator.train(training_data)
    results[model_type] = estimator.evaluate(test_data)
```

## Dataset

The project utilizes comprehensive BMS data with the following characteristics:

- **Size**: 30+ GB of raw numerical data
- **Duration**: 4 months of continuous recording
- **Frequency**: Microsecond-precision measurements
- **Parameters**: 
  - Bidirectional current measurements
  - Cell and pack voltage readings
  - Temperature variations
  - Charge/discharge cycle data
  - Timestamped SoC values

### Data Structure
```
├── voltage/          # Cell and pack voltage measurements
├── current/          # Bidirectional current flow data
├── temperature/      # Thermal measurement data
├── soc_reference/    # BMS-measured SoC values
└── metadata/         # Timestamp and cycle information
```

## Models

### Model 1: ANN + XGBoost
- **Architecture**: 4 dense layers with ReLU activation
- **Regularization**: 3 dropout layers
- **Training**: 170 epochs with early stopping
- **Performance**: MSE > 500

### Model 2: ANN + AdaBoost
- **Architecture**: Similar to XGBoost model
- **Training**: 270 epochs
- **Performance**: MSE 400-500 (best performing)
- **Advantage**: Stays within SoC bounds (0-100%)

### Model 3: ANN + Bayesian Ridge
- **Architecture**: Probabilistic approach
- **Training**: 310 epochs
- **Performance**: MSE > 800
- **Challenge**: Early stopping due to regularization

## Results

### Performance Comparison
| Model | MSE | Epochs | SoC Boundary Adherence | Generalization |
|-------|-----|--------|----------------------|----------------|
| XGBoost | >500 | 170 | ±25% overshoot | Good |
| AdaBoost | 400-500 | 270 | Excellent | Best |
| Bayesian Ridge | >800 | 310 | Poor | Limited |

### Key Findings
- **AdaBoost** showed the best overall performance
- **XGBoost** improved with time steps >1.2 seconds
- All models struggled with extreme SoC values (0% and 100%)
- Traditional Coulomb counting remained competitive in certain scenarios

## Contributing

We welcome contributions to improve the SoC estimation algorithms:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Create a Pull Request

### Areas for Contribution
- Model architecture optimization
- Additional battery chemistry support
- Real-time implementation improvements
- Enhanced data preprocessing techniques

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **University of Bath** - Department of Electronic & Electrical Engineering
- **Author**: Aseem Saxena

## References

For detailed technical information and citations, please refer to the full research paper and bibliography available in the repository.

## 🔗 Links

- [Research Paper](./State_of_Charge_Algorithm_on_a_LiFePO4_and_NMC_battery_pack.pdf)
- [GitHub Repository](https://github.com/AseemLab)
- [University of Bath EEE Department](https://www.bath.ac.uk/departments/department-of-electronic-electrical-engineering/)

---

**Note**: This project represents academic research in battery management systems. For production applications, additional validation and testing are recommended.
