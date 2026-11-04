# Dynamic Visualizations Guide
## Live 2026 Predictor - Interactive Graphs

### Overview

The Live 2026 Predictor tab now includes **5 dynamic, interactive visualizations** that update in real-time based on user inputs. These graphs provide deeper insights into the prediction logic and help stakeholders understand market dynamics.

---

## 📊 Visualization Components

### 1️⃣ Price Forecast Comparison (Bar Chart)

**What it shows:**
- Current price (gray baseline)
- Predictions from all 3 models (XGBoost, BiLSTM, Ridge)
- Percentage change from current price for each model

**Dynamic behavior:**
- Bars adjust height based on predicted prices
- Color-coded by model (XGBoost=red, BiLSTM=teal, Ridge=blue)
- Shows absolute price (₹) and percentage change

**Use case:**
- Quickly compare model predictions
- Identify consensus or divergence
- Visual confidence in forecast direction

**Example interpretation:**
```
Current: ₹2100
XGBoost: ₹2352 (+12%)  ← Most aggressive
BiLSTM: ₹2310 (+10%)   ← Middle estimate
Ridge: ₹2268 (+8%)     ← Most conservative
```

---

### 2️⃣ Sentiment Impact Gauge

**What it shows:**
- VADER sentiment score (-1 to +1 scale)
- Color-coded zones:
  - Red zone (-1 to -0.25): Negative sentiment
  - Yellow zone (-0.25 to +0.25): Neutral
  - Green zone (+0.25 to +1): Positive sentiment
- Red threshold line at -0.25 (disruption trigger)

**Dynamic behavior:**
- Needle moves based on news headline input
- Delta shows change from neutral (0)
- Caption updates with interpretation

**Use case:**
- Assess news/weather sentiment impact
- Identify if sentiment crosses shock threshold
- Understand media narrative influence

**Example scenarios:**
```
Calm: "Good supply" → +0.62 (green) → "Market stability"
Shock: "Crop damage" → -0.91 (red) → "Negative sentiment detected"
```

---

### 3️⃣ Weather Risk Assessment Gauge

**What it shows:**
- Composite weather risk score (0-100%)
- Calculated from:
  - Rainfall > 15mm: +50%
  - Temperature > 42°C: +30%
  - Temperature < 20°C: +20%
- Color zones: Green (0-33%), Yellow (33-66%), Red (66-100%)

**Dynamic behavior:**
- Updates based on rainfall and temperature inputs
- Caption shows specific values triggering risk

**Use case:**
- Quick visual weather stress indicator
- Combines multiple weather factors
- Flags extreme conditions

**Example calculations:**
```
Calm: 0mm rain, 36°C → 0% risk (green)
Moderate: 20mm rain → 50% risk (yellow)
High: 30mm rain + 44°C → 80% risk (red)
```

---

### 4️⃣ Simulated Price Trajectory (7-Day Projection)

**What it shows:**
- 3 scenario lines over next 7 days:
  - **Optimistic** (dotted blue): 75% of forecast change rate
  - **Most Likely** (solid teal): Model forecast trend
  - **Pessimistic** (dashed red): 125% of forecast change (amplified if shock detected)
- Horizontal reference line at current price

**Dynamic behavior:**
- All curves start at current price
- Slope determined by forecast vs current price gap
- Divergence widens if shock is detected
- X-axis shows actual dates (e.g., "Mar 15", "Mar 16")

**Use case:**
- Visualize forward price trajectory
- Plan inventory/procurement decisions
- Assess range of possible outcomes

**Example interpretation:**
```
Day 0: ₹2100 (current)
Day 1: ₹2121 (optimistic) | ₹2158 (likely) | ₹2195 (pessimistic)
Day 7: ₹2180 (optimistic) | ₹2352 (likely) | ₹2524 (pessimistic)
```

**Note:** Disclaimer shown below chart clarifying this is a simulation, not accounting for future events.

---

### 5️⃣ Feature Contribution Analysis (Radar Chart)

**What it shows:**
- 5 feature categories on radar axes:
  1. **Price Lag** (Recent History): Distance of lag-1 from current price
  2. **Volatility** (7-Day Std): Recent price fluctuation
  3. **Sentiment** (News + Weather): Absolute VADER score
  4. **Weather** (Rainfall): Rainfall normalized to 0-1 scale
  5. **Intraday Spread**: Max-Min price range

**Dynamic behavior:**
- Each axis scaled 0-100%
- Larger polygon area = Greater overall feature influence
- All values normalized for comparison

**Use case:**
- Understand which factors dominate the prediction
- Identify primary drivers (e.g., weather vs sentiment)
- Explain model behavior to non-technical stakeholders

**Example shapes:**
```
Calm scenario:
  - Small, symmetric pentagon (all features ~20-30%)
  - No dominant factor
  
Shock scenario:
  - Large, asymmetric shape
  - Sentiment and Rainfall axes extended (>80%)
  - Clear visual dominance of external factors
```

---

## 💡 Key Insights Summary Section

**What it shows:**
Automatically generated bullet-point insights based on:
- Price change magnitude and direction
- Sentiment score thresholds
- Weather extremes
- Model consensus/divergence
- Shock probability levels

**Dynamic rules:**

| Condition | Insight Generated |
|-----------|-------------------|
| Forecast > +10% | 🔴 "Significant price surge predicted" |
| Forecast +5% to +10% | 🟡 "Moderate upward pressure" |
| Forecast -5% to +5% | 🟢 "Stable price dynamics" |
| Forecast < -5% | 🔵 "Price decline expected" |
| Sentiment < -0.5 | 📰 "Very negative news sentiment" |
| Sentiment > +0.5 | 📰 "Positive news sentiment" |
| Rainfall > 25mm | 🌧️ "Heavy rainfall alert" |
| Rainfall 15-25mm | 🌦️ "Unseasonal rain detected" |
| Temp > 42°C | 🌡️ "Heatwave conditions" |
| Model spread < 3% | 🤝 "High model consensus" |
| Model spread > 3% | ⚠️ "Model divergence detected" |
| Shock prob > 75% | 🚨 "Critical shock probability" |
| Shock prob 50-75% | ⚠️ "Elevated shock risk" |

**Use case:**
- Quick executive summary
- Policy recommendations
- Stakeholder briefing points

---

## 🎯 Testing the Visualizations

### Test Scenario 1: Calm Market

**Inputs:**
```
Modal Price: ₹2100
Rainfall: 0mm
Temperature: 36°C
Headline: "Good supply conditions"
```

**Expected Visualizations:**
- 📊 Bar chart: All models within ±2% of current price
- 📈 Sentiment gauge: Green zone (~+0.6)
- 🌡️ Weather gauge: 0% risk (green)
- 📉 Trajectory: Flat, slowly rising lines
- 🕸️ Radar: Small, symmetric pentagon
- 💡 Insights: "Stable price dynamics", "Market equilibrium"

### Test Scenario 2: Supply Shock

**Inputs:**
```
Modal Price: ₹2100
Rainfall: 25mm
Temperature: 36°C
Headline: "Heavy downpour damages crops"
```

**Expected Visualizations:**
- 📊 Bar chart: Models predict +8% to +12% surge
- 📈 Sentiment gauge: Red zone (-0.44), below -0.25 threshold
- 🌡️ Weather gauge: 50% risk (yellow/red)
- 📉 Trajectory: Steep upward divergence, pessimistic line accelerates
- 🕸️ Radar: Large shape, Sentiment and Rainfall axes extended
- 💡 Insights: "Significant price surge", "Heavy rainfall alert", "Critical shock probability"

### Test Scenario 3: Heatwave

**Inputs:**
```
Modal Price: ₹1800
Rainfall: 0mm
Temperature: 44°C
Headline: "Extreme heat affecting crops"
```

**Expected Visualizations:**
- 📊 Bar chart: Moderate upward predictions (+5-8%)
- 📈 Sentiment gauge: Slightly negative (-0.2 to -0.3)
- 🌡️ Weather gauge: 30% risk (yellow) from temperature alone
- 📉 Trajectory: Gradual upward slope
- 🕸️ Radar: Weather axis extended, others moderate
- 💡 Insights: "Heatwave conditions", "Crop stress and storage concerns"

---

## 🛠️ Technical Implementation

### Libraries Used
- **Plotly.js**: All interactive charts (bar, gauge, line, radar)
- **Streamlit**: Layout and rendering
- **Pandas/NumPy**: Data manipulation

### Performance
- All graphs render in <200ms
- No external API calls (fully client-side)
- Responsive design (adapts to screen size)

### Code Location
- File: `app.py`
- Function: `tab_live_predictor()`
- Lines: ~450-700 (after Model Comparison table)

---

## 📈 Benefits for Viva Demonstration

1. **Interactivity:** Reviewers can change inputs and see immediate graph updates
2. **Interpretability:** Complex ML predictions become visually intuitive
3. **Stakeholder Communication:** Non-technical users understand drivers
4. **Confidence Building:** Multiple views of same prediction increase trust
5. **Professional Presentation:** Publication-quality interactive visualizations

---

## 🎓 Viva Q&A Preparation

**Q: Why 5 graphs instead of just showing the prediction number?**  
**A:** Each graph serves a distinct purpose:
- Bar chart: Model comparison
- Sentiment gauge: External signal strength
- Weather gauge: Environmental risk
- Trajectory: Forward-looking uncertainty
- Radar: Feature decomposition

Together they provide 360° view of the prediction.

**Q: How do the graphs respond to user inputs?**  
**A:** All graphs are dynamically generated using Plotly with data from the user's form submission. No fixed data—every chart reflects the specific scenario the user creates. For example, changing rainfall from 0mm to 25mm will instantly update the weather gauge, trajectory slope, and radar chart.

**Q: What if models disagree wildly?**  
**A:** The bar chart will show large spread, and the "Key Insights" section will flag "Model divergence detected". The trajectory chart will show wider fan-out between scenarios, visually indicating uncertainty.

**Q: Can these graphs be exported for reports?**  
**A:** Yes, Plotly charts have built-in export functionality (camera icon). Users can download as PNG for presentations or PDF reports.

---

**End of Dynamic Visualizations Guide**
