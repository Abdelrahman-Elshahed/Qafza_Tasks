import logging
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pandas as pd
import joblib

@transformer
def train(df: pd.DataFrame):
    try:
        # Feature Engineering
        df['price'] = np.log1p(df['price'])  # Log transform price
        
        # Create new features
        df['reviews_to_listings_ratio'] = df['number_of_reviews'] / df['calculated_host_listings_count']
        df['reviews_per_month_filled'] = df['reviews_per_month'].fillna(0)
        df['days_since_last_review_filled'] = df['days_since_last_review'].fillna(365)
        
        # Remove outliers
        q1 = df['price'].quantile(0.01)
        q3 = df['price'].quantile(0.99)
        df = df[(df['price'] >= q1) & (df['price'] <= q3)]
        
        # Identify features
        numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        numeric_features = [col for col in numeric_features if col != 'price']
        
        # Create preprocessing pipeline
        numeric_transformer = Pipeline(steps=[
            ('scaler', RobustScaler())
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features)
            ])

        # Split data
        X = df.drop('price', axis=1)
        y = df['price']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Define models with better hyperparameters
        models = {
            "Random Forest": RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            ),
            "XGBoost": XGBRegressor(
                n_estimators=200,
                max_depth=7,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            ),
            "LightGBM": LGBMRegressor(
                n_estimators=200,
                num_leaves=31,
                learning_rate=0.1,
                feature_fraction=0.8,
                random_state=42
            )
        }

        results = {}
        
        # Train and evaluate each model
        for model_name, model in models.items():
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('regressor', model)
            ])
            
            # Train
            pipeline.fit(X_train, y_train)
            
            # Predict
            preds = pipeline.predict(X_test)
            
            # Convert predictions back from log scale
            preds = np.expm1(preds)
            y_test_orig = np.expm1(y_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test_orig, preds)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test_orig, preds)
            
            # Log results
            logging.info(f"{model_name} Metrics:")
            logging.info(f"  RMSE: {rmse:.2f}")
            logging.info(f"  R² Score: {r2:.4f}")
            
            # Feature importance (for Random Forest)
            if model_name == "Random Forest":
                feature_imp = pd.DataFrame(
                    model.feature_importances_,
                    index=numeric_features,
                    columns=['importance']
                ).sort_values('importance', ascending=False)
                logging.info(f"\nFeature Importance:\n{feature_imp}")
            
            results[model_name] = {
                "MSE": float(mse),
                "RMSE": float(rmse),
                "R²": float(r2)
            }
            
            # Save model
            joblib.dump({
                'pipeline': pipeline,
                'metrics': results[model_name],
                'features': numeric_features
            }, f'{model_name.lower().replace(" ", "_")}_model.pkl')
        
        return results
        
    except Exception as e:
        logging.error(f"An error occurred during training: {e}")
        raise