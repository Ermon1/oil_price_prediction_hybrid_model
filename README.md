# Oil Price Prediction API

A hybrid ARIMA-LSTM model for oil price forecasting, implemented as a FastAPI service.

## Features

- Hybrid ARIMA-LSTM model for improved prediction accuracy
- FastAPI REST API with automatic documentation
- Real-time predictions with confidence scores
- Health check endpoint for monitoring
- Data preprocessing and feature engineering

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/oil_price_prediction.git
cd oil_price_prediction
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
python -m app.main
```

The API will be available at `http://localhost:8000`

## API Documentation

Once the server is running, you can access:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## API Endpoints

- `POST /api/v1/predict/json/`: Make predictions using JSON data
- `GET /health`: Check system health and model status

## Project Structure

```
oil_price_prediction/
├── app/
│   ├── api/
│   │   ├── core/
│   │   └── endpoints/
│   ├── ml/
│   │   ├── model.py
│   │   └── preprocessing.py
│   └── main.py
├── tests/
├── requirements.txt
└── README.md
```

## License

MIT License
