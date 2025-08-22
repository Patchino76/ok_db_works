#%% Simple Regime Analysis using Rolling Statistics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from get_dataframe import load_data
from scipy import stats
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')

# Configuration
TARGET_FEATURES = ['Ore', 'WaterMill', 'WaterZumpf', 'PulpHC', 'PumpRPM', 'MotorAmp', 'PSI200']
MIN_REGIME_DURATION = 180  # 3 hours in minutes
RESAMPLE_FREQ = '5T'

class SimpleRegimeAnalyzer:
    """Simple regime detection using rolling statistics and change detection"""
    
    def __init__(self, min_regime_duration=MIN_REGIME_DURATION):
        self.min_regime_duration = min_regime_duration
        self.data = None
        self.regimes = {}
        
    def load_data(self):
        """Load and prepare data"""
        print(f"📊 Loading data with {RESAMPLE_FREQ} resampling...")
        self.data = load_data(resample=RESAMPLE_FREQ)
        available_features = [f for f in TARGET_FEATURES if f in self.data.columns]
        self.data = self.data[available_features].dropna()
        print(f"✅ Data loaded: {len(self.data)} samples, {len(available_features)} features")
        print(f"📅 Date range: {self.data.index.min()} to {self.data.index.max()}")
        return self.data
    
    def detect_changepoints_simple(self, feature, window=30, threshold=2.0):
        """Simple changepoint detection using rolling statistics"""
        data = self.data[feature].values
        
        # Calculate rolling mean and std
        rolling_mean = pd.Series(data).rolling(window=window, center=True).mean()
        rolling_std = pd.Series(data).rolling(window=window, center=True).std()
        
        # Calculate z-scores for change detection
        z_scores = np.abs((data - rolling_mean) / rolling_std)
        
        # Find peaks in z-scores (potential changepoints)
        peaks, _ = find_peaks(z_scores, height=threshold, distance=window//2)
        
        return peaks
    
    def extract_regimes_simple(self, feature):
        """Extract regimes from changepoints"""
        changepoints = self.detect_changepoints_simple(feature)
        
        # Add start and end points
        boundaries = np.concatenate([[0], changepoints, [len(self.data)-1]])
        boundaries = np.unique(boundaries)
        
        regimes = []
        for i in range(len(boundaries)-1):
            start_idx = boundaries[i]
            end_idx = boundaries[i+1]
            
            start_time = self.data.index[start_idx]
            end_time = self.data.index[end_idx]
            duration_minutes = (end_time - start_time).total_seconds() / 60
            
            if duration_minutes >= self.min_regime_duration:
                regime_data = self.data.iloc[start_idx:end_idx+1][feature]
                
                regimes.append({
                    'regime_id': len(regimes),
                    'start_time': start_time,
                    'end_time': end_time,
                    'start_idx': start_idx,
                    'end_idx': end_idx,
                    'duration_hours': duration_minutes / 60,
                    'mean_value': regime_data.mean(),
                    'std_value': regime_data.std(),
                    'sample_count': len(regime_data)
                })
        
        self.regimes[feature] = regimes
        print(f"📋 Found {len(regimes)} regimes for {feature}")
        return regimes
    
    def plot_regimes(self, feature, figsize=(15, 6)):
        """Plot regimes for a feature"""
        if feature not in self.regimes:
            self.extract_regimes_simple(feature)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot time series
        ax.plot(self.data.index, self.data[feature], 'k-', alpha=0.7, linewidth=0.8)
        
        # Color regimes
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.regimes[feature])))
        
        for i, regime in enumerate(self.regimes[feature]):
            start_time = regime['start_time']
            end_time = regime['end_time']
            
            ax.axvspan(start_time, end_time, alpha=0.3, color=colors[i], 
                      label=f"Regime {regime['regime_id']} ({regime['duration_hours']:.1f}h)")
            
            # Add regime info
            mid_time = start_time + (end_time - start_time) / 2
            ax.text(mid_time, regime['mean_value'], 
                   f"R{regime['regime_id']}\n{regime['mean_value']:.1f}±{regime['std_value']:.1f}",
                   ha='center', va='bottom', fontsize=8,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor=colors[i], alpha=0.7))
        
        ax.set_ylabel(feature)
        ax.set_title(f'Simple Regime Analysis: {feature} - {len(self.regimes[feature])} Regimes')
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        return fig
    
    def analyze_all_features(self):
        """Analyze all features"""
        results = {}
        
        for feature in TARGET_FEATURES:
            if feature in self.data.columns:
                try:
                    regimes = self.extract_regimes_simple(feature)
                    results[feature] = {
                        'regimes': regimes,
                        'total_regimes': len(regimes)
                    }
                except Exception as e:
                    print(f"❌ Error analyzing {feature}: {e}")
                    results[feature] = {'error': str(e)}
        
        return results
    
    def create_summary(self):
        """Create summary DataFrame"""
        summary_data = []
        
        for feature, regimes in self.regimes.items():
            for regime in regimes:
                summary_data.append({
                    'Feature': feature,
                    'Regime_ID': regime['regime_id'],
                    'Start_Time': regime['start_time'],
                    'End_Time': regime['end_time'],
                    'Duration_Hours': regime['duration_hours'],
                    'Mean_Value': regime['mean_value'],
                    'Std_Value': regime['std_value'],
                    'Sample_Count': regime['sample_count']
                })
        
        return pd.DataFrame(summary_data)

def main():
    """Main analysis function"""
    print("🚀 Starting Simple Regime Analysis")
    print("="*50)
    
    # Initialize analyzer
    analyzer = SimpleRegimeAnalyzer()
    
    # Load data
    try:
        data = analyzer.load_data()
        print(f"✅ Data loaded successfully: {data.shape}")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Analyze all features
    print("\n🔬 Analyzing regimes for all features...")
    results = analyzer.analyze_all_features()
    
    # Create summary
    summary = analyzer.create_summary()
    
    # Print results
    print("\n" + "="*50)
    print("📋 REGIME ANALYSIS SUMMARY")
    print("="*50)
    
    for feature in TARGET_FEATURES:
        if feature in results and 'total_regimes' in results[feature]:
            total_regimes = results[feature]['total_regimes']
            print(f"• {feature:12}: {total_regimes:2d} regimes")
    
    # Save summary
    output_file = Path(__file__).parent / 'simple_regime_summary.csv'
    summary.to_csv(output_file, index=False)
    print(f"\n💾 Summary saved to: {output_file}")
    
    # Create plots
    print("\n🎨 Creating plots...")
    plots_dir = Path(__file__).parent / 'simple_regime_plots'
    plots_dir.mkdir(exist_ok=True)
    
    for feature in TARGET_FEATURES:
        if feature in analyzer.regimes and len(analyzer.regimes[feature]) > 0:
            print(f"📊 Plotting {feature}...")
            fig = analyzer.plot_regimes(feature)
            
            plot_file = plots_dir / f'simple_regime_{feature}.png'
            fig.savefig(plot_file, dpi=300, bbox_inches='tight')
            print(f"💾 Saved: {plot_file}")
            plt.close(fig)  # Close to save memory
    
    print("\n✅ Simple regime analysis complete!")
    print(f"📁 Plots saved in: {plots_dir}")
    print(f"📄 Summary saved as: {output_file}")
    
    return analyzer, results, summary

if __name__ == "__main__":
    try:
        analyzer, results, summary = main()
        print("\n🎉 Analysis completed successfully!")
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
