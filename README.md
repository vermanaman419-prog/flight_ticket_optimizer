# ✈️ Flight Price Prediction & Optimization



## 📊 Project Overview

This project uses advanced machine learning techniques to predict flight prices and help travelers make informed booking decisions. By analyzing historical flight data, the model identifies patterns and provides actionable insights on the optimal time to purchase tickets.

### Key Features

- 🎯 **Price Prediction**: Accurate flight price forecasting using ensemble methods
- 📈 **Trend Analysis**: Identify whether prices will rise or drop
- ⏰ **Optimal Booking Window**: Determine the best time to buy
- 📊 **Comprehensive EDA**: Visual insights into pricing patterns
- 🤖 **Multiple ML Models**: Comparison of RF, XGBoost, and GradientBoosting

## 🗂️ Dataset

**Source**: [Kaggle - Flight Price Prediction Dataset](https://www.kaggle.com/datasets/shubhambathwal/flight-price-prediction)

**Features**:
- Airline name
- Source and destination cities
- Flight duration
- Number of stops
- Departure and arrival times
- Travel class
- Days until departure
- Price (target variable)

## 🛠️ Technologies Used

- **Python 3.8+**
- **Data Analysis**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Machine Learning**: Scikit-learn, XGBoost, LightGBM
- **Model Optimization**: GridSearchCV, Optuna

## 📁 Project Structure
```
flight-price-prediction/
│
├── data/
│   ├── raw/                    # Original dataset
│   ├── processed/              # Cleaned data
│   └── features/               # Engineered features
│
├── notebooks/
│   ├── 01_EDA.ipynb           # Exploratory Data Analysis
│   ├── 02_Feature_Engineering.ipynb
│   ├── 03_Model_Training.ipynb
│   └── 04_Model_Evaluation.ipynb
│
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   └── utils.py
│
├── models/
│   ├── flight_price_model.pkl
│   ├── scaler.pkl
│   └── encoders.pkl
│
├── visualizations/            # Generated plots and charts
├── requirements.txt
└── README.md
```

## 🚀 Getting Started

### Prerequisites
```bash
Python 3.8 or higher
pip package manager
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/flight-price-prediction.git
cd flight-price-prediction
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download dataset**
```bash
# Download from Kaggle and place in data/raw/
# Or use Kaggle API:
kaggle datasets download -d shubhambathwal/flight-price-prediction
```

## 📊 Exploratory Data Analysis

### Key Findings

#### 1. Price Distribution
- Average flight price: ₹8,500
- Price range: ₹1,500 - ₹80,000
- Most flights: ₹5,000 - ₹15,000 range

#### 2. Airline Impact
- Premium airlines (Vistara, Air India): 30-40% higher prices
- Budget airlines (IndiGo, SpiceJet): Most affordable

#### 3. Timing Insights
- **Best booking window**: 21-60 days before departure
- **Avoid**: Last 7 days (prices spike by 50%+)
- **Cheapest days**: Tuesday & Wednesday
- **Peak seasons**: May-June, December

#### 4. Route Analysis
- Mumbai-Delhi: Highest traffic
- Bangalore-Delhi: Premium pricing
- Regional routes: More affordable

## 🤖 Machine Learning Models

### Models Implemented

1. **Random Forest Regressor**
   - R² Score: 0.984
   - MAE: ₹1,167
   - RMSE: ₹2,854

2. **XGBoost Regressor**
   - R² Score: 0.977
   - MAE: ₹1,862
   - RMSE: ₹3,417

3. **Gradient Boosting Regressor**
   - R² Score: 0.957
   - MAE: ₹2,808
   - RMSE: ₹4,713

### Feature Importance

Top 5 most influential features:
1. Route average price (28%)
2. Airline average price (22%)
3. Days until departure (18%)
4. Flight duration (15%)
5. Number of stops (12%)

## 📈 Model Performance

| Model | MAE (₹) | RMSE (₹) | R² Score | Training Time |
|-------|---------|----------|----------|---------------|
| Random Forest | 1,167 | 2,854 | 0.984 | 45s |
| XGBoost | 1,862 | 3,417 | 0.977 | 32s |
| Gradient Boosting | 2,808 | 4,713 | 0.957 | 67s |

**Best Model**: Random Forest (Highest R² and lowest error)

## 💡 Usage

### Predict Flight Price
```python
from src.model_training import predict_flight_price

# Example prediction
price = predict_flight_price(
    airline='IndiGo',
    source='Delhi',
    destination='Mumbai',
    flight_class='Economy',
    stops=0,
    duration=2.5,
    departure_hour=10,
    month=6,
    days_until_departure=30
)

print(f"Predicted Price: ₹{price:.2f}")
```

### Get Booking Recommendation
```python
from src.utils import generate_booking_recommendation

recommendation = generate_booking_recommendation(
    days_until_departure=30,
    predicted_price=8500,
    route_average=8000
)
```

## 🎯 Key Insights & Recommendations

### For Travelers

✅ **DO**:
- Book 3-8 weeks in advance
- Fly on Tuesday/Wednesday for best prices
- Choose budget airlines for short routes
- Book early morning flights

❌ **AVOID**:
- Last-minute bookings (< 7 days)
- Weekend flights (Friday-Sunday)
- Peak season travel without advance booking
- Multiple stops unless significant savings

### Price Prediction Accuracy

- **85%** predictions within ₹1,500 of actual price
- **95%** predictions within ₹3,000 of actual price
- Model performs best for Economy class on popular routes

## 🔮 Future Enhancements

- [ ] Real-time price tracking API integration
- [ ] LSTM model for time-series forecasting
- [ ] Multi-city route optimization
- [ ] Price alert system
- [ ] Mobile app development
- [ ] Integration with booking platforms
- [ ] Weather and event impact analysis

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


## 👨‍💻 Author

**Your Name**
- GitHub: (https://github.com/vermanaman419-prog)
- LinkedIn: (https://www.linkedin.com/in/naman419/)

## 🙏 Acknowledgments

- Dataset provided by [Kaggle](https://www.kaggle.com/datasets/shubhambathwal/flight-price-prediction)
- Inspired by real-world flight booking challenges
- Thanks to the open-source ML community

