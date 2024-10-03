# MLOps-PumpMaintenance-Dashboard using Streamlit | Docker | AWS |


This project demonstrates a predictive maintenance system for industrial pumps using Streamlit, Docker, AWS.

## Setup

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Generate synthetic data: `python src/data_generator.py`
4. Train the model: `python src/train_model.py`
5. Run the Streamlit app: `streamlit run app.py`

## Docker Deployment

To deploy using Docker:

1. Build the Docker image: `docker build -t predictive-maintenance .`
2. Run the Docker container: `docker run -p 8501:8501 predictive-maintenance`

The app will be available at http://localhost:8501

## AWS EC2 Deployment ( for reference, watch this video : https://www.youtube.com/watch?v=qNIniDftAcU )

1. Launch an EC2 instance (t2.micro for free tier)
2. Install Docker on the EC2 instance
3. Copy your project files to the EC2 instance
4. Build and run the Docker container as described above
5. Configure security group to allow inbound traffic on port 8501

I would like to thank Vincent Stevenson for his video on Hosting a docker on AWS EC2 instance.
