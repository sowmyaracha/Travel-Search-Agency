# Travel-Search-Agency
# Travel Booking Prediction Model

A machine learning project to predict hotel booking conversions from search data.

---

## Project Overview

**Goal:** Predict whether a user will book a hotel listing based on their search behavior

**Problem Type:** Binary Classification (Imbalanced Data)

**Dataset:** Travel search and booking data with 54 features, ~74,000 samples

**Target Variable:** `booked` (0 = Not Booked, 1 = Booked)

---

## Key Features

- **Multiple ML Algorithms:** Random Forest, Gradient Boosting, Logistic Regression
- **Handles Class Imbalance:** Uses SMOTE for intelligent oversampling
- **Prevents Data Leakage:** Removes future-information features
- **Feature Engineering:** Temporal features from timestamps
- **Proper Validation:** Train/test split with cross-validation
- **Production-Ready:** Clean pipeline for deployment

---

### Prerequisites
```bash
pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn
```

### Run the Model
```python
# 1. Open the notebook
travel_booking_CLEAN.ipynb

# 2. Update the file path (cell 2)
df = pd.read_csv('your_path/train_small.csv')

# 3. Run all cells (Runtime > Run all)

# 4. View results in the final cells
```

---

## 🔧 Pipeline Overview

```
1. Load Data
   ↓
2. Remove Leaky Features (clicked, booking_value)
   ↓
3. Feature Engineering (timestamp → hour, day, weekend)
   ↓
4. Handle Missing Values
   ↓
5. Train/Test Split (80/20)
   ↓
6. Feature Scaling (StandardScaler)
   ↓
7. Balance Training Data (SMOTE)
   ↓
8. Train Models (RF, GBM, LR)
   ↓
9. Evaluate on Test Set
   ↓
10. Select Best Model
```

---

## 📊 Data Description

### Input Features (~50)
- **Search Information:** timestamp, site_id, user_country_id
- **Listing Details:** price, stars, location, review_score
- **User History:** historical_stars, historical_paid
- **Competitor Data:** competitor prices, availability
- **Temporal Features:** hour, day_of_week, month, is_weekend

### Target Variable
- **booked:** 1 if user booked the listing, 0 otherwise
- **Class Distribution:** ~5% booked, 95% not booked (highly imbalanced)

### Removed Features (Data Leakage)
- **clicked:** User action AFTER search
- **booking_value:** Known only AFTER booking

---

## Key Concepts

### Data Leakage Prevention
**Problem:** Including features that contain future information leads to 100% accuracy in testing but 0% in production.

**Solution:**
- Remove all features known only AFTER the target event
- Split data BEFORE any preprocessing
- Fit preprocessing on training data only

### Class Imbalance Handling
**Problem:** Only 5% of searches result in bookings.

**Solution:**
- Use SMOTE to create synthetic minority class examples
- Apply only to training data
- Evaluate on original test distribution

### Feature Scaling
**Problem:** Features have different scales (price: $50-500, stars: 1-5).

**Solution:**
- StandardScaler normalizes all features
- Fit on training data only
- Critical for Logistic Regression and neural networks

---

## Model Details

### Random Forest
```python
RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)
```
- **Pros:** Handles non-linear relationships, resistant to overfitting
- **Cons:** Slower prediction time
- **Best For:** High accuracy, feature importance analysis

### Gradient Boosting
```python
GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)
```
- **Pros:** Often highest accuracy, good with imbalanced data
- **Cons:** Slower training, prone to overfitting
- **Best For:** Maximum predictive performance

### Logistic Regression
```python
LogisticRegression(
    max_iter=1000,
    random_state=42,
    n_jobs=-1
)
```
- **Pros:** Fast, interpretable, probability calibration
- **Cons:** Linear decision boundary
- **Best For:** Fast predictions, baseline model

---

## Evaluation Metrics

### Why Multiple Metrics?

**Accuracy:** Not reliable for imbalanced data
- With 95% not booked, predicting all "not booked" = 95% accuracy!

**Precision:** Of predicted bookings, how many were correct?
- Important to avoid wasting resources on false positives

**Recall:** Of actual bookings, how many did we catch?
- Important to not miss potential customers

**F1-Score:** Harmonic mean of Precision and Recall
- Best single metric for imbalanced data

**ROC-AUC:** Area under ROC curve
- Measures ability to distinguish between classes
- 0.5 = random, 1.0 = perfect

---

## Common Issues & Solutions

### Issue 1: 100% Accuracy
**Cause:** Data leakage (features with future information)
**Solution:** Use `travel_booking_NO_LEAKAGE.ipynb`

### Issue 2: Low Accuracy (<50%)
**Cause:** Insufficient features or poor quality data
**Solution:** 
- Add more features
- Better feature engineering
- Collect more data

### Issue 3: High Overfitting (Train >> Test)
**Cause:** Model too complex
**Solution:**
- Reduce max_depth
- Increase min_samples_split
- Add regularization

### Issue 4: Poor Recall on Bookings
**Cause:** Class imbalance
**Solution:**
- Increase SMOTE sampling_strategy
- Adjust class_weight
- Try different threshold

---

## Feature Importance

Top features typically include:
1. **price** - Listing price
2. **listing_stars** - Hotel rating
3. **user_hist_stars** - User's typical preference
4. **listing_review_score** - Review rating
5. **hour** - Time of search
6. **is_weekend** - Weekend vs weekday
7. **competitor prices** - Price comparison
8. **day_of_week** - Day pattern
9. **month** - Seasonal effect
10. **user_country_id** - Geographic preference

---

## Production Deployment

### Step 1: Train Final Model
```python
# Use all available data
final_model = RandomForestClassifier(...)
final_model.fit(X_all_scaled, y_all)
```

### Step 2: Save Model
```python
import joblib
joblib.dump(final_model, 'booking_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
```

### Step 3: Load and Predict
```python
model = joblib.load('booking_model.pkl')
scaler = joblib.load('scaler.pkl')

# New search data
X_new_scaled = scaler.transform(X_new)
predictions = model.predict_proba(X_new_scaled)[:, 1]
```

### Step 4: Set Threshold
```python
# Adjust threshold based on business needs
threshold = 0.3  # Lower = more bookings predicted
bookings = predictions > threshold
```

---

## Business Impact

### Typical Use Cases
1. **Ranking:** Show listings most likely to be booked first
2. **Pricing:** Adjust prices based on booking probability
3. **Promotions:** Target users unlikely to book with discounts
4. **Inventory:** Predict demand for capacity planning

### Example ROI
```
Scenario: 100,000 searches/day, 5% booking rate = 5,000 bookings

With Model (70% accuracy):
- Better targeting → 1% conversion improvement
- 100 extra bookings/day
- $100 average booking value
- $10,000/day = $3.65M/year revenue increase
```

---

## Model Improvement Roadmap

### Phase 1: Baseline (Current)
- Clean data pipeline
- Multiple algorithms
- Proper validation
- 65-75% accuracy

### Phase 2: Feature Engineering
- [ ] User session features
- [ ] Price ratio features
- [ ] Competitive index
- [ ] User behavior clusters
- **Target:** 70-80% accuracy

### Phase 3: Advanced Models
- [ ] XGBoost with tuning
- [ ] Neural networks
- [ ] Ensemble stacking
- [ ] Time-series features
- **Target:** 75-85% accuracy

### Phase 4: Production Optimization
- [ ] A/B testing
- [ ] Online learning
- [ ] Real-time predictions
- [ ] Monitoring & alerts

---

## Additional Resources

### Documentation
- `DATA_LEAKAGE_EXPLAINED.md` - Complete guide on data leakage
- `ACCURACY_IMPROVEMENT_GUIDE.md` - How to boost performance
- `BEFORE_AFTER_COMPARISON.md` - Performance improvements

### Notebooks
- `travel_booking_CLEAN.ipynb` - Production version (use this)
- `travel_booking_NO_LEAKAGE.ipynb` - Educational version with validation
- `travel_booking_ADVANCED.ipynb` - Extended version with XGBoost

### Quick References
- `DATA_LEAKAGE_QUICK_REFERENCE.md` - Cheat sheet

---

## Troubleshooting

### Error: "FileNotFoundError"
```python
# Update the file path in cell 2
df = pd.read_csv('/your/actual/path/train_small.csv')
```

### Error: "ModuleNotFoundError: imbalanced-learn"
```bash
pip install imbalanced-learn
```

### Error: "ValueError: Input contains NaN"
- Data has missing values
- Check data quality
- Increase threshold for dropping columns

### Warning: "Test accuracy > 95%"
- Likely data leakage
- Check for removed features
- Verify train/test split order

---

## Support

### Common Questions

**Q: Why is accuracy only 70% and not 95%+?**
A: 70% is realistic for this problem. 95%+ usually indicates data leakage.

**Q: Can I add more features?**
A: Yes, but ensure they're available at prediction time (no future information).

**Q: Why remove 'clicked' and 'booking_value'?**
A: These happen AFTER the search, so they wouldn't be available when making predictions.

**Q: How do I improve accuracy?**
A: See `ACCURACY_IMPROVEMENT_GUIDE.md` for detailed strategies.

---

## License

This project is for educational purposes.

---

## Acknowledgments

- Dataset: Travel booking search data
- Libraries: scikit-learn, imbalanced-learn, pandas, numpy
- Techniques: SMOTE, feature engineering, proper validation

---

## Version History

### v3.0 (Clean Production)
- Removed all unnecessary code
- Streamlined pipeline
- Production-ready notebook

### v2.0 (No Data Leakage)
- Fixed 100% accuracy issue
- Proper train/test split
- Removed leaky features

### v1.0 (Initial)
- Basic Naive Bayes model
- 8.48% accuracy
- Had data leakage issues

---

## Quick Summary

**Problem:** Predict hotel bookings (5% booking rate)

**Solution:** Random Forest with SMOTE on cleaned data

**Performance:** 70-75% accuracy, 0.75-0.80 ROC-AUC

**Key Success:** Prevented data leakage, handled imbalance properly

**Production Ready:** Yes, deployable pipeline

---

**Ready to predict bookings! **
