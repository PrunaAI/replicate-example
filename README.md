# Flux Dev: Replicate Example

Flux Dev is an example project that demonstrates how to build, optimize, and deploy a deep learning model using Cog and Pruna AI's model smashing (optimization) tools. This repository is configured to work with GPU support and integrates with Replicate for model hosting and inference.

> **Note:** This repository leverages Cog for model building and prediction, and uses a GitHub Actions workflow for pushing the model to Replicate. Ensure you have the necessary tokens and credentials configured to run and deploy the model.

---

## Repository Structure

prunaai-replicate-example/  
├── README.md  
├── flux-dev/  
│   ├── cog.yaml       (Cog configuration file for building and running the model)  
│   └── predict.py     (Predictor implementation with model smashing and caching setup)  
└── .github/  
    └── workflows/  
        └── push_flux_dev.yaml  (GitHub Actions workflow to push the model to Replicate)

---

## Getting Started

### Prerequisites

- **Docker:** Required for building and running Cog environments.
- **Git:** For version control and repository management.
- **Cog CLI:** Install the Cog CLI from https://github.com/replicate/cog to build and run the model locally.
- **Replicate API Token:** (Optional) Set up your Replicate API token in GitHub Secrets (as REPLICATE_API_TOKEN) if you plan to push the model automatically using GitHub Actions.

### Installation
1. **Clone the repository:**  
   Run the following commands in your terminal:  
   ```bash
   git clone https://github.com/your-username/prunaai-replicate-example.git  
   cd replicate-example
   ```
2. **Configure Environment Variables:**  
   - For local testing, ensure that you have Docker running.  
   - Replace `<your-token>` in the file flux-dev/predict.py with your actual token. If you don't have a token, you can get one from here: https://docs.pruna.ai/en/stable/setup/pip.html#installing-pruna-pro

3. **Install Cog (if you haven't already):**  
   ```bash
   pip install cog
   ```

---

## Building and Running the Model Locally

The model is defined in the flux-schnell directory and is built using the cog.yaml configuration.

1. **Build the Model:**  
   Navigate to the flux-dev directory and build the model using Cog:  
   ```bash
   cd flux-dev  
   cog build
   ```

   This process installs the required system packages (like libgl1-mesa-glx, git, build-essential), Python dependencies, and sets up the environment with GPU support.

2. **Run a Prediction:**  
   After building the model, you can run a prediction. For example, execute:  
   ```bash
   cog run --input prompt="a scenic landscape with mountains" --input num_inference_steps=28 --input guidance_scale=7.5
   ```

   This command performs the following actions in the predict function of predict.py:
   - Loads the Flux model using the FluxPipeline.
   - Optimizes the model with Pruna AI's smash tool using caching configurations.
   - Generates an image based on the provided prompt and other inference parameters.
   - Saves the output image to a temporary directory.

---

## GitHub Actions Workflow for Replicate

The repository includes a GitHub Actions workflow located at .github/workflows/push_flux_dev.yaml that automates pushing the built model to Replicate.

### How It Works

- **Workflow Trigger:**  
  The workflow can be triggered manually using the "workflow_dispatch" event from the GitHub Actions tab. You can specify a custom model name if desired; if left blank, the model name defaults based on the image value in cog.yaml.

- **Steps in the Workflow:**  
  1. **Free Disk Space:** Cleans up disk space to ensure sufficient room for model building.  
  2. **Checkout Repository:** Fetches the latest code from the repository.  
  3. **Setup Cog:** Installs Docker buildx and Cog, optionally handling CUDA.  
  4. **Push to Replicate:** Executes cog push from the flux-schnell directory to deploy the model to Replicate.

### Pushing to Replicate Manually

If you need to push the model manually, navigate to the flux-schnell directory and run:  

```bash
cog push r8.im/prunaai/flux-dev  
```

Ensure that your Replicate API token is set either in GitHub Secrets or in your local environment.

---

## Customization

- **Model Optimization:**  
  The model optimization is configured in flux-dev/predict.py using the SmashConfig settings. Modify these parameters to suit your use case.

- **Dependencies:**  
  Update or add Python packages as needed by modifying the commands in the run section of cog.yaml.

- **Hardware & CUDA:**  
  If you require a different CUDA version or additional system packages, update the corresponding fields in cog.yaml.

---

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests if you have suggestions or improvements.

1. Fork the repository.
2. Create a new branch for your changes.
3. Submit a Pull Request with a detailed description of your modifications.

---

## License

This project is licensed under the MIT License.

---

## Contact

For any questions or further details, please open an issue or join our [Discord community](https://discord.com/invite/rskEr4BZJx).

---

Happy coding and enjoy optimizing your models with Replicate and Pruna AI!