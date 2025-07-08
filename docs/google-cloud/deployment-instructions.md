# Deploying FastAPI App to Google Cloud Run

This document provides detailed instructions for deploying the `pytest-fastapi-template` FastAPI application to Google Cloud Run, a managed platform for containerized applications.

## Prerequisites

Before starting, ensure you have the following:
- A Google Cloud Platform (GCP) account with a project set up. If not, create one at [Google Cloud Console](https://console.cloud.google.com/).
- Billing enabled for your GCP project (required for Cloud Run deployment).
- Docker installed on your local machine. If not, download it from [Docker's official site](https://www.docker.com/products/docker-desktop).
- The Google Cloud SDK (`gcloud`) installed and authenticated. If not installed, follow the instructions at [Google Cloud SDK Installation](https://cloud.google.com/sdk/docs/install). After installation, authenticate with `gcloud auth login`.

## Deployment Steps

### Step 1: Containerize Your FastAPI Application

Your existing `Dockerfile` is already configured for deployment:
- It uses `python:3.11-slim` as the base image.
- It installs dependencies from `requirements.txt`.
- It copies the application code and sets up the working directory.
- It exposes a configurable port via the `PORT` environment variable.
- It runs the app with `uvicorn`, binding to `0.0.0.0` and using the `PORT` variable (defaulting to 8000 if not set).

No changes are needed to the `Dockerfile` for Cloud Run compatibility.

### Step 2: Build and Push the Docker Image

You need to build the Docker image and push it to Google Container Registry (GCR).

1. **Set up Google Cloud SDK**: If not already done, authenticate with `gcloud auth login`.
2. **Configure Docker for GCR**: Authenticate Docker with GCR by running:
   ```bash
   gcloud auth configure-docker
   ```
3. **Build the Docker Image**: Tag it for GCR and build it. Replace `your-project-id` with your Google Cloud project ID.
   ```bash
   docker build -t gcr.io/your-project-id/fastapi-app:latest .
   ```
4. **Push the Image to GCR**:
   ```bash
   docker push gcr.io/your-project-id/fastapi-app:latest
   ```

### Step 3: Deploy to Cloud Run

Deploy the container to Cloud Run with the following command. Replace `your-project-id` with your actual Google Cloud project ID and adjust the region if needed (e.g., `us-central1`).

```bash
gcloud run deploy fastapi-service \
  --image gcr.io/your-project-id/fastapi-app:latest \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

- `--allow-unauthenticated` makes the service publicly accessible. If your API requires authentication, remove this flag and set up IAM permissions or API keys.
- Cloud Run will automatically set the `PORT` environment variable to 8080, which your `Dockerfile`'s `uvicorn` command will use via `${PORT:-8000}`.

### Step 4: Set Environment Variables

Your application uses environment variables for API keys (e.g., `elevenlabs_api_key`, `openai_api_key`). Set these in Cloud Run with:

```bash
gcloud run deploy fastapi-service \
  --image gcr.io/your-project-id/fastapi-app:latest \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars "elevenlabs_api_key=YOUR_ELEVENLABS_KEY,openai_api_key=YOUR_OPENAI_KEY"
```

Alternatively, you can set environment variables via the Cloud Run console after deployment.

### Step 5: Test the Deployment

After deployment, Cloud Run will provide a service URL (e.g., `https://fastapi-service-XXXXX-uc.a.run.app`). You can test your endpoints, such as:
- `GET /v1/hello` to see the "Hello World" message.
- `POST /v1/gemini/podcast` for podcast generation (ensure API keys are set as environment variables in Cloud Run).

## Additional Configuration

- **Scaling**: Cloud Run automatically scales your application based on traffic, with a minimum instance count of 0 (cost-effective when idle). You can configure scaling settings in the Cloud Run console.
- **Custom Domain**: If you want a custom domain, map it in the Cloud Run console under the "Custom Domains" tab after deployment.
- **CI/CD**: Consider setting up a CI/CD pipeline with Cloud Build to automate future deployments. Create a `cloudbuild.yaml` file with build and deploy steps, then trigger builds via `gcloud builds submit`.

## Troubleshooting

- **Deployment Errors**: Check logs in the Cloud Run console or with `gcloud run services logs read fastapi-service --region us-central1`.
- **API Key Issues**: Ensure environment variables are correctly set if endpoints return authentication errors.
- **Docker Push Errors**: Verify `gcloud auth configure-docker` was successful and your project ID is correct.

## Resources

- [Google Cloud Run Documentation](https://cloud.google.com/run/docs)
- [Deploying to Cloud Run](https://cloud.google.com/run/docs/deploying)
- [Setting Environment Variables in Cloud Run](https://cloud.google.com/run/docs/configuring/environment-variables)

If you encounter issues or need further assistance, refer to the Google Cloud support resources or contact your development team.
