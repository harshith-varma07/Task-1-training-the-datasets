import pandas as pd
import numpy as np
from typing import Union, Literal, Tuple, Optional, List
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, 
    LabelEncoder, OneHotEncoder
)
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
import warnings
warnings.filterwarnings('ignore')


class DatasetPreprocessor:
    def __init__(self, data): 
        if isinstance(data, str):
            self.df = pd.read_csv(data)
        elif isinstance(data, pd.DataFrame):
            self.df = data.copy()
        else:
            raise ValueError("Data must be a DataFrame or path to CSV file")
        self.original_shape = self.df.shape
        self.numeric_columns = []
        self.categorical_columns = []
        self.scalers = {}
        self.encoders = {}
        
    def analyze_dataset(self):
        
        print("=" * 80)
        print("DATASET ANALYSIS")
        print("=" * 80)
        print(f"\nOriginal Shape: {self.df.shape}")
        print(f"Rows: {self.df.shape[0]}, Columns: {self.df.shape[1]}")
        
        print("\n" + "-" * 80)
        print("DATA TYPES")
        print("-" * 80)
        print(self.df.dtypes)
        
        print("\n" + "-" * 80)
        print("MISSING VALUES")
        print("-" * 80)
        missing = self.df.isnull().sum()
        missing_percent = (missing / len(self.df)) * 100
        missing_df = pd.DataFrame({
            'Missing Count': missing,
            'Percentage': missing_percent
        })
        print(missing_df[missing_df['Missing Count'] > 0])
        
        print("\n" + "-" * 80)
        print("BASIC STATISTICS")
        print("-" * 80)
        print(self.df.describe())
        
        print("\n" + "-" * 80)
        print("DUPLICATE ROWS")
        print("-" * 80)
        duplicates = self.df.duplicated().sum()
        print(f"Number of duplicate rows: {duplicates}")
        
        return self.df.info()
    
    def identify_column_types(self):
        
        self.numeric_columns = self.df.select_dtypes(
            include=['int64', 'float64', 'int32', 'float32']
        ).columns.tolist()
        
        self.categorical_columns = self.df.select_dtypes(
            include=['object', 'category']
        ).columns.tolist()
        
        print(f"\nNumeric columns ({len(self.numeric_columns)}): {self.numeric_columns}")
        print(f"Categorical columns ({len(self.categorical_columns)}): {self.categorical_columns}")
        
        return self.numeric_columns, self.categorical_columns
    
    def remove_duplicates(self, subset: Optional[List[str]] = None, 
                         keep: Literal['first', 'last', False] = 'first') -> pd.DataFrame:
        
        initial_rows = len(self.df)
        self.df = self.df.drop_duplicates(subset=subset, keep=keep)
        removed = initial_rows - len(self.df)
        print(f"Removed {removed} duplicate rows")
        return self.df
    
    def handle_missing_values(self, strategy='auto', numeric_strategy='mean', 
                            categorical_strategy='most_frequent', threshold=0.5):
        
        print("\n" + "=" * 80)
        print("HANDLING MISSING VALUES")
        print("=" * 80)
        
        if strategy == 'auto':
            # Drop columns with too many missing values
            missing_ratio = self.df.isnull().sum() / len(self.df)
            cols_to_drop = missing_ratio[missing_ratio > threshold].index.tolist()
            if cols_to_drop:
                print(f"Dropping columns with >{threshold*100}% missing: {cols_to_drop}")
                self.df = self.df.drop(columns=cols_to_drop)
            
            # Impute remaining missing values
            self.identify_column_types()
            strategy = 'impute'
        
        if strategy == 'impute':
            # Impute numeric columns
            if self.numeric_columns:
                numeric_imputer = SimpleImputer(strategy=numeric_strategy)
                self.df[self.numeric_columns] = numeric_imputer.fit_transform(
                    self.df[self.numeric_columns]
                )
                print(f"Imputed numeric columns using '{numeric_strategy}' strategy")
            
            # Impute categorical columns
            if self.categorical_columns:
                categorical_imputer = SimpleImputer(strategy=categorical_strategy)
                self.df[self.categorical_columns] = categorical_imputer.fit_transform(
                    self.df[self.categorical_columns]
                )
                print(f"Imputed categorical columns using '{categorical_strategy}' strategy")
        
        elif strategy == 'drop':
            initial_rows = len(self.df)
            self.df = self.df.dropna()
            removed = initial_rows - len(self.df)
            print(f"Dropped {removed} rows with missing values")
        
        print(f"Remaining missing values: {self.df.isnull().sum().sum()}")
        return self.df
    
    def handle_outliers(self, method='iqr', threshold=1.5, columns=None):
        
        print("\n" + "=" * 80)
        print("HANDLING OUTLIERS")
        print("=" * 80)
        
        if columns is None:
            columns = self.numeric_columns
        
        initial_rows = len(self.df)
        
        for col in columns:
            if col not in self.df.columns:
                continue
                
            if method == 'iqr':
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                outliers = ((self.df[col] < lower_bound) | (self.df[col] > upper_bound)).sum()
                if outliers > 0:
                    print(f"{col}: {outliers} outliers detected")
                    # Cap outliers instead of removing
                    self.df[col] = self.df[col].clip(lower=lower_bound, upper=upper_bound)
            
            elif method == 'zscore':
                z_scores = np.abs((self.df[col] - self.df[col].mean()) / self.df[col].std())
                outliers = (z_scores > threshold).sum()
                if outliers > 0:
                    print(f"{col}: {outliers} outliers detected")
                    # Remove rows with z-score > threshold
                    self.df = self.df[z_scores <= threshold]
        
        removed = initial_rows - len(self.df)
        if method == 'iqr':
            print(f"Outliers capped using IQR method (threshold={threshold})")
        else:
            print(f"Removed {removed} rows with outliers using z-score method (threshold={threshold})")
        
        return self.df
    
    def encode_categorical_variables(self, method='auto', columns=None):
        
        print("\n" + "=" * 80)
        print("ENCODING CATEGORICAL VARIABLES")
        print("=" * 80)
        
        if columns is None:
            columns = self.categorical_columns
        
        for col in columns:
            if col not in self.df.columns:
                continue
            
            unique_values = self.df[col].nunique()
            
            # Auto-select encoding method
            if method == 'auto':
                encoding_method = 'label' if unique_values > 10 else 'onehot'
            else:
                encoding_method = method
            
            if encoding_method == 'label':
                # Label Encoding
                le = LabelEncoder()
                self.df[col] = le.fit_transform(self.df[col].astype(str))
                self.encoders[col] = le
                print(f"{col}: Label encoded ({unique_values} unique values)")
            
            elif encoding_method == 'onehot':
                # One-Hot Encoding
                dummies = pd.get_dummies(self.df[col], prefix=col, drop_first=True)
                self.df = pd.concat([self.df, dummies], axis=1)
                self.df = self.df.drop(columns=[col])
                print(f"{col}: One-hot encoded ({unique_values} unique values -> {len(dummies.columns)} columns)")
        
        # Update column types
        self.identify_column_types()
        return self.df
    
    def normalize_features(self, method='standard', columns=None):
        
        print("\n" + "=" * 80)
        print("NORMALIZING FEATURES")
        print("=" * 80)
        
        if columns is None:
            columns = self.numeric_columns
        
        if not columns:
            print("No numeric columns to normalize")
            return self.df
        
        # Select scaler
        if method == 'standard':
            scaler = StandardScaler()
            print("Using StandardScaler (mean=0, std=1)")
        elif method == 'minmax':
            scaler = MinMaxScaler()
            print("Using MinMaxScaler (range 0-1)")
        elif method == 'robust':
            scaler = RobustScaler()
            print("Using RobustScaler (robust to outliers)")
        else:
            raise ValueError("Method must be 'standard', 'minmax', or 'robust'")
        
        # Fit and transform
        self.df[columns] = scaler.fit_transform(self.df[columns])
        self.scalers[method] = scaler
        
        print(f"Normalized {len(columns)} columns: {columns}")
        return self.df
    
    def feature_selection(self, target_column, n_features=10, task='classification'):
        
        print("\n" + "=" * 80)
        print("FEATURE SELECTION")
        print("=" * 80)
        
        if target_column not in self.df.columns:
            print(f"Error: Target column '{target_column}' not found")
            return self.df
        
        X = self.df.drop(columns=[target_column])
        y = self.df[target_column]
        score_func = f_classif if task == 'classification' else f_regression
        selector = SelectKBest(score_func=score_func, k=min(n_features, X.shape[1]))
        selector.fit(X, y)
        selected_features = X.columns[selector.get_support()].tolist()
        
        print(f"Selected top {len(selected_features)} features:")
        for i, feature in enumerate(selected_features, 1):
            print(f"  {i}. {feature}")
        self.df = self.df[selected_features + [target_column]]
        return self.df
    
    def split_dataset(self, target_column, test_size=0.2, random_state=42, 
                     validation_split=False, val_size=0.1):
        
        print("\n" + "=" * 80)
        print("SPLITTING DATASET")
        print("=" * 80)
        
        if target_column not in self.df.columns:
            raise ValueError(f"Target column '{target_column}' not found")
        
        X = self.df.drop(columns=[target_column])
        y = self.df[target_column]
        
        if validation_split:
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            
            # Split train+val into train and val
            val_ratio = val_size / (1 - test_size)
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=val_ratio, random_state=random_state
            )
            
            print(f"Training set: {X_train.shape[0]} samples")
            print(f"Validation set: {X_val.shape[0]} samples")
            print(f"Test set: {X_test.shape[0]} samples")
            print(f"Features: {X_train.shape[1]}")
            
            return X_train, X_val, X_test, y_train, y_val, y_test
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            
            print(f"Training set: {X_train.shape[0]} samples")
            print(f"Test set: {X_test.shape[0]} samples")
            print(f"Features: {X_train.shape[1]}")
            
            return X_train, X_test, y_train, y_test
    
    def export_processed_data(self, filepath, include_index=False):
        
        self.df.to_csv(filepath, index=include_index)
        print(f"\n✓ Processed data exported to: {filepath}")
        print(f"  Shape: {self.df.shape}")
        return filepath
    
    def get_processed_data(self):
       
        return self.df
    
    def full_preprocessing_pipeline(self, target_column=None, 
                                   normalize_method='standard',
                                   encoding_method='auto',
                                   handle_outliers_method='iqr',
                                   missing_strategy='auto',
                                   split_data=False,
                                   export_path=None):
        
        print("\n" + "=" * 80)
        print("FULL PREPROCESSING PIPELINE")
        print("=" * 80)
        
        # Step 1: Analyze
        self.analyze_dataset()
        
        # Step 2: Identify column types
        self.identify_column_types()
        
        # Step 3: Remove duplicates
        self.remove_duplicates()
        
        # Step 4: Handle missing values
        self.handle_missing_values(strategy=missing_strategy)
        
        # Step 5: Handle outliers
        if self.numeric_columns:
            self.handle_outliers(method=handle_outliers_method)
        
        # Step 6: Encode categorical variables
        if self.categorical_columns:
            self.encode_categorical_variables(method=encoding_method)
        
        # Step 7: Normalize features
        if self.numeric_columns:
            self.normalize_features(method=normalize_method)
        
        # Step 8: Export if requested
        if export_path:
            self.export_processed_data(export_path)
        
        # Step 9: Split data if requested
        if split_data:
            if target_column is None:
                raise ValueError("target_column must be specified when split_data=True")
            return self.split_dataset(target_column)
        
        print("\n" + "=" * 80)
        print("PREPROCESSING COMPLETE")
        print("=" * 80)
        print(f"Final shape: {self.df.shape}")
        
        return self.df


# Example usage and demonstration
def example_usage():
    
    print("EXAMPLE USAGE OF DATASET PREPROCESSOR")
    print("=" * 80)
    
    # Create a sample dataset
    np.random.seed(42)
    sample_data = pd.DataFrame({
        'age': np.random.randint(18, 80, 1000),
        'income': np.random.normal(50000, 15000, 1000),
        'credit_score': np.random.randint(300, 850, 1000),
        'gender': np.random.choice(['M', 'F', 'Other'], 1000),
        'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], 1000),
        'employed': np.random.choice(['Yes', 'No'], 1000),
        'loan_approved': np.random.choice([0, 1], 1000)
    })
    
    # Add some missing values
    sample_data.loc[np.random.choice(sample_data.index, 50), 'income'] = np.nan
    sample_data.loc[np.random.choice(sample_data.index, 30), 'education'] = np.nan
    
    # Add some duplicates
    sample_data = pd.concat([sample_data, sample_data.iloc[:10]], ignore_index=True)
    
    print("\nCreated sample dataset with 1010 rows (including duplicates)")
    print("Added missing values in 'income' and 'education' columns")
    
    # Initialize preprocessor
    preprocessor = DatasetPreprocessor(sample_data)
    
    # Run full pipeline
    processed_data = preprocessor.full_preprocessing_pipeline(
        target_column='loan_approved',
        normalize_method='standard',
        encoding_method='auto',
        handle_outliers_method='iqr',
        missing_strategy='auto',
        split_data=False,
        export_path=None  # Set to 'processed_data.csv' to export
    )
    
    print("\n✓ Pipeline complete! You can now:")
    print("  1. Get processed data: preprocessor.get_processed_data()")
    print("  2. Split data: preprocessor.split_dataset('loan_approved')")
    print("  3. Export data: preprocessor.export_processed_data('output.csv')")
    
    return preprocessor


if __name__ == "__main__":
    # Run example
    example_usage()
    
    print("\n" + "=" * 80)
    print("TO USE WITH YOUR OWN DATA:")
    print("=" * 80)
    print("""
# Load your data
preprocessor = DatasetPreprocessor('your_data.csv')

# Option 1: Run full automated pipeline
preprocessor.full_preprocessing_pipeline(
    target_column='your_target_column',
    normalize_method='standard',
    split_data=True,
    export_path='processed_data.csv'
)

# Option 2: Step-by-step processing
preprocessor.analyze_dataset()
preprocessor.identify_column_types()
preprocessor.remove_duplicates()
preprocessor.handle_missing_values(strategy='auto')
preprocessor.handle_outliers(method='iqr')
preprocessor.encode_categorical_variables(method='auto')
preprocessor.normalize_features(method='standard')

# Get the processed data
processed_df = preprocessor.get_processed_data()

# Split into train/test sets
X_train, X_test, y_train, y_test = preprocessor.split_dataset('target_column')
    """)