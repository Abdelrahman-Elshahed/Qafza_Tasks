import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

@transformer
def transform(data, *args, **kwargs):
    """
    Template code for a transformer block.
    Args:
        data: The output from the upstream parent block (DataFrame)
        args: The output from any additional upstream blocks (if applicable)
    Returns:
        Transformed DataFrame after cleaning and feature engineering.
    """
    # تنظيف البيانات
    data = data.drop(['id', 'name', 'host_id', 'host_name', 'license'], axis=1)
    
    # معالجة القيم الناقصة
    data['reviews_per_month'] = data['reviews_per_month'].fillna(0)
    
    # تحويل التواريخ
    data['last_review'] = pd.to_datetime(data['last_review'])
    data['days_since_last_review'] = (pd.Timestamp.now() - data['last_review']).dt.days
    
    # تحديد الأعمدة الفئوية والعددية
    categorical_features = ['neighbourhood_group', 'room_type']
    numeric_features = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # إنشاء المعالج
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numeric_features),
            ('cat', OneHotEncoder(drop='first'), categorical_features)
        ]
    )
    
    # تطبيق المعالج
    transformed_data = preprocessor.fit_transform(data)
    
    # تحويل النتيجة إلى DataFrame
    transformed_columns = numeric_features + list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features))
    transformed_df = pd.DataFrame(transformed_data, columns=transformed_columns)
    
    return transformed_df