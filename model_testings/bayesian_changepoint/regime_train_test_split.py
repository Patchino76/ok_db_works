#!/usr/bin/env python3
"""
Regime-based Train/Test Split for XGBoost Model
Creates training and testing datasets using regime labels from simple_regime_analysis.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from get_dataframe import load_data
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class RegimeTrainTestSplitter:
    """Create train/test splits based on regime analysis results"""
    
    def __init__(self, test_size=0.2, random_state=42):
        self.test_size = test_size
        self.random_state = random_state
        self.data = None
        self.regime_summary = None
        self.labeled_data = None
        
    def load_data_and_regimes(self):
        """Load original data and regime analysis results"""
        print("📊 Loading original data...")
        self.data = load_data(resample='5T')  # Match the regime analysis resampling
        print(f"✅ Data loaded: {self.data.shape}")
        
        # Load regime summary
        regime_file = Path(__file__).parent / 'simple_regime_summary.csv'
        if not regime_file.exists():
            raise FileNotFoundError(f"Regime summary not found: {regime_file}")
        
        print("📋 Loading regime analysis results...")
        self.regime_summary = pd.read_csv(regime_file)
        self.regime_summary['Start_Time'] = pd.to_datetime(self.regime_summary['Start_Time'])
        self.regime_summary['End_Time'] = pd.to_datetime(self.regime_summary['End_Time'])
        
        print(f"✅ Regime summary loaded: {len(self.regime_summary)} regimes")
        return self.data, self.regime_summary
    
    def add_regime_labels(self, feature='PSI200'):
        """Add regime labels to the original dataframe"""
        print(f"🏷️  Adding regime labels for {feature}...")
        
        # Filter regimes for the specified feature
        feature_regimes = self.regime_summary[self.regime_summary['Feature'] == feature].copy()
        
        if feature_regimes.empty:
            raise ValueError(f"No regimes found for feature {feature}")
        
        # Create a copy of the data with regime labels
        self.labeled_data = self.data.copy()
        self.labeled_data['regime_id'] = -1  # Default: no regime
        self.labeled_data['regime_mean'] = np.nan
        self.labeled_data['regime_std'] = np.nan
        self.labeled_data['regime_duration'] = np.nan
        
        # Assign regime labels based on time periods
        for _, regime in feature_regimes.iterrows():
            mask = (
                (self.labeled_data.index >= regime['Start_Time']) & 
                (self.labeled_data.index <= regime['End_Time'])
            )
            
            self.labeled_data.loc[mask, 'regime_id'] = regime['Regime_ID']
            self.labeled_data.loc[mask, 'regime_mean'] = regime['Mean_Value']
            self.labeled_data.loc[mask, 'regime_std'] = regime['Std_Value']
            self.labeled_data.loc[mask, 'regime_duration'] = regime['Duration_Hours']
        
        # Remove data points not assigned to any regime
        self.labeled_data = self.labeled_data[self.labeled_data['regime_id'] != -1].copy()
        
        print(f"✅ Regime labels added: {len(self.labeled_data)} samples across {len(feature_regimes)} regimes")
        print(f"📊 Regime distribution:")
        regime_counts = self.labeled_data['regime_id'].value_counts().sort_index()
        for regime_id, count in regime_counts.items():
            print(f"   Regime {regime_id}: {count:,} samples")
        
        return self.labeled_data
    
    def create_feature_engineering(self):
        """Skip feature engineering - use only original features"""
        print("🔧 Skipping feature engineering - using original features only...")
        
        print(f"✅ Using original features: {self.labeled_data.shape[1]} features")
        return self.labeled_data
    
    def create_train_test_split(self, target_column='PSI200'):
        """Create regime-level train/test split preserving temporal structure"""
        print("🎯 Creating regime-level train/test split...")
        
        # Get unique regimes and their sizes
        regime_info = self.labeled_data.groupby('regime_id').agg({
            target_column: 'count',
            'regime_duration': 'first'
        }).rename(columns={target_column: 'sample_count'})
        
        print(f"📊 Available regimes:")
        for regime_id, info in regime_info.iterrows():
            print(f"   Regime {regime_id}: {info['sample_count']:,} samples ({info['regime_duration']:.1f}h)")
        
        # Shuffle regimes (not individual samples)
        np.random.seed(self.random_state)
        regime_ids = list(regime_info.index)
        np.random.shuffle(regime_ids)
        
        # Split regimes to achieve approximately the desired test_size
        total_samples = len(self.labeled_data)
        target_test_samples = int(total_samples * self.test_size)
        
        test_regimes = []
        test_sample_count = 0
        
        # Greedily assign regimes to test set until we reach target size
        for regime_id in regime_ids:
            regime_samples = regime_info.loc[regime_id, 'sample_count']
            if test_sample_count + regime_samples <= target_test_samples * 1.2:  # Allow 20% tolerance
                test_regimes.append(regime_id)
                test_sample_count += regime_samples
            if test_sample_count >= target_test_samples * 0.8:  # Stop when we have at least 80% of target
                break
        
        train_regimes = [r for r in regime_ids if r not in test_regimes]
        
        print(f"📊 Regime assignment:")
        print(f"   Train regimes: {train_regimes}")
        print(f"   Test regimes: {test_regimes}")
        
        # Create train/test datasets by regime assignment
        train_mask = self.labeled_data['regime_id'].isin(train_regimes)
        test_mask = self.labeled_data['regime_id'].isin(test_regimes)
        
        train_data = self.labeled_data[train_mask].copy()
        test_data = self.labeled_data[test_mask].copy()
        
        # Sort by timestamp to maintain temporal order within each dataset
        train_data = train_data.sort_index()
        test_data = test_data.sort_index()
        
        print(f"✅ Split created (preserving temporal structure):")
        print(f"   Training set: {len(train_data):,} samples ({len(train_data)/len(self.labeled_data)*100:.1f}%)")
        print(f"   Test set: {len(test_data):,} samples ({len(test_data)/len(self.labeled_data)*100:.1f}%)")
        
        # Show regime distribution in splits
        print(f"📊 Regime distribution in splits:")
        train_regime_counts = train_data['regime_id'].value_counts().sort_index()
        test_regime_counts = test_data['regime_id'].value_counts().sort_index()
        
        for regime_id in sorted(self.labeled_data['regime_id'].unique()):
            if regime_id in train_regimes:
                count = train_regime_counts.get(regime_id, 0)
                print(f"   Regime {regime_id}: TRAIN - {count:,} samples")
            else:
                count = test_regime_counts.get(regime_id, 0)
                print(f"   Regime {regime_id}: TEST - {count:,} samples")
        
        return train_data, test_data
    
    def save_datasets(self, train_data, test_data, output_dir=None):
        """Save train and test datasets to CSV files with only features and target"""
        if output_dir is None:
            output_dir = Path(__file__).parent
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(exist_ok=True)
        
        # Filter to keep only features and target (exclude regime metadata)
        exclude_cols = ['regime_id', 'regime_mean', 'regime_std', 'regime_duration']
        feature_cols = [col for col in train_data.columns if col not in exclude_cols]
        
        train_clean = train_data[feature_cols].copy()
        test_clean = test_data[feature_cols].copy()
        
        # Save datasets
        train_file = output_dir / 'regime_train_data.csv'
        test_file = output_dir / 'regime_test_data.csv'
        
        print("💾 Saving datasets (features and target only)...")
        train_clean.to_csv(train_file, index=True)
        test_clean.to_csv(test_file, index=True)
        
        print(f"✅ Datasets saved:")
        print(f"   Training data: {train_file}")
        print(f"   Test data: {test_file}")
        
        # Save metadata
        metadata = {
            'total_samples': len(train_data) + len(test_data),
            'train_samples': len(train_data),
            'test_samples': len(test_data),
            'test_size': self.test_size,
            'random_state': self.random_state,
            'features': list(train_data.columns),
            'regimes_count': len(train_data['regime_id'].unique()),
            'date_range': f"{train_data.index.min()} to {train_data.index.max()}"
        }
        
        metadata_file = output_dir / 'regime_split_metadata.json'
        import json
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        print(f"   Metadata: {metadata_file}")
        
        return train_file, test_file, metadata_file

def main():
    """Main function to create regime-based train/test split"""
    print("🚀 Starting Regime-based Train/Test Split")
    print("=" * 50)
    
    try:
        # Initialize splitter
        splitter = RegimeTrainTestSplitter(test_size=0.2, random_state=42)
        
        # Load data and regimes
        data, regime_summary = splitter.load_data_and_regimes()
        
        # Add regime labels
        labeled_data = splitter.add_regime_labels(feature='PSI200')
        
        # Create engineered features
        enhanced_data = splitter.create_feature_engineering()
        
        # Create train/test split
        train_data, test_data = splitter.create_train_test_split(
            target_column='PSI200'
        )
        
        # Save datasets
        train_file, test_file, metadata_file = splitter.save_datasets(train_data, test_data)
        
        print("\n" + "=" * 50)
        print("✅ Regime-based train/test split completed successfully!")
        print(f"📁 Files created:")
        print(f"   • {train_file}")
        print(f"   • {test_file}")
        print(f"   • {metadata_file}")
        
        # Summary statistics
        print(f"\n📊 Summary:")
        print(f"   • Total samples: {len(train_data) + len(test_data):,}")
        print(f"   • Training samples: {len(train_data):,}")
        print(f"   • Test samples: {len(test_data):,}")
        print(f"   • Features: {len([col for col in train_data.columns if col not in ['PSI200', 'regime_id', 'regime_mean', 'regime_std', 'regime_duration']])}")
        print(f"   • Regimes: {len(train_data['regime_id'].unique())}")
        
        return splitter, train_data, test_data
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None

if __name__ == "__main__":
    splitter, train_data, test_data = main()
