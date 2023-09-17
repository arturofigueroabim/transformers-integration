# Research Overview

## Objective
This study aims to answer the research question:

> "Which transformer-based background knowledge integration strategies exist currently and how well do they perform in direct comparison on a standard downstream argument mining task such as argument relation classification?"

We focus on transformer-based background knowledge integration strategies for argument relation classification in argument mining. The strategies under evaluation include , ERNIE, KEPLER, and LIBERT.

## Importance
Understanding the effectiveness of these strategies is crucial to:
- Identify state-of-the-art approaches to background knowledge integration.
- Enhance the performance of existing models.
- Guide future research in argument mining.

## Benefits
Through this study, we will:
- Understand the strengths and weaknesses of different strategies.
- Provide insights to improve model performances.
- Contribute to the advancement of argument mining by identifying effective strategies.

## Goals
The study hopes to:
- Develop more effective models for argument mining.
- Provide insights on transformer-based background knowledge integration strategies.
- Have practical implications for developing accurate systems in argument mining.

# Project Setup

## Tool Configuration: Poetry
This project uses Poetry for dependency management and packaging.

## Dependencies
- Python: ^3.10
- Torch: 2.0.0
- Transformers: ^4.30.2
- Datasets: ^2.13.0
- PEFT: ^0.3.0
- Lightning: ^2.0.3
- JupyterLab: ^4.0.2
- IPyWidgets: ^8.0.6
- TQDM: ^4.65.0
- Sentencepiece: ^0.1.99
- WandB: ^0.15.4
- TikToken: ^0.4.0
- Scikit-Learn: ^1.3.0
- Pydantic: ^2.0.2
- PyTorch-Lightning: ^2.0.4
- Matplotlib: ^3.7.2
- Seaborn: ^0.12.2

## Running the Project
To evaluate the models using the provided script:

1. Ensure that the configuration file (`config.py`) has been set up correctly and that any paths to datasets, models, or other resources are correct.
2. Run the evaluation script using:
```bash
python evaluate.py  
```

1. Ensure you have all required libraries installed. If using Poetry, you can install all dependencies via:
```bash
poetry install
```

## Author
Arturo Figueroa

## Acknowledgments
I would like to thank everyone who contributed to the open-source libraries used in this project and the supervisors who guided me through this research journey.
