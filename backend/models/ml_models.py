import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

class MLModelManager:
    """Manages ML models for warehouse slotting optimization."""
    
    def __init__(self):
        self.available_models = {
            "random_forest": "Random Forest - Assigns locations based on product characteristics",
            "collaborative_filtering": "Collaborative Filtering - Groups similar products together",
            "ranking": "Ranking Algorithm - Assigns by pick frequency priority"
        }
        self.label_encoders = {}
        self.scaler = StandardScaler()
    
    def get_available_models(self) -> dict:
        """Get list of available ML models."""
        return self.available_models
    
    def apply_model(self, df: pd.DataFrame, model_name: str) -> dict:
        """Apply selected ML model to optimize locations."""
        try:
            if df.empty:
                return {"success": False, "message": "Empty dataframe provided"}
            
            if model_name not in self.available_models:
                return {"success": False, "message": f"Model '{model_name}' not available"}
            
            # Validate data
            validation_result = self._validate_input_data(df)
            if not validation_result["success"]:
                return validation_result
            
            # Apply the selected model
            if model_name == "random_forest":
                result = self._apply_random_forest(df)
            elif model_name == "collaborative_filtering":
                result = self._apply_collaborative_filtering(df)
            elif model_name == "ranking":
                result = self._apply_ranking_algorithm(df)
            else:
                return {"success": False, "message": f"Model implementation not found: {model_name}"}
            
            return result
            
        except Exception as e:
            return {"success": False, "message": f"Error applying model: {str(e)}"}
    
    def _validate_input_data(self, df: pd.DataFrame) -> dict:
        """Validate input data for ML processing."""
        try:
            required_columns = ['SKU', 'Location', 'QTY', 'Line', 'Location_Zone']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                return {"success": False, "message": f"Missing required columns: {missing_columns}"}
            
            if len(df) < 5:
                return {"success": False, "message": "Insufficient data for optimization (minimum 5 rows required)"}
            
            if df['Line'].sum() == 0:
                return {"success": False, "message": "No pick frequency data available"}
            
            return {"success": True, "message": "Data validation successful"}
            
        except Exception as e:
            return {"success": False, "message": f"Validation error: {str(e)}"}
    
    def _apply_random_forest(self, df: pd.DataFrame) -> dict:
        """Apply Random Forest model for location optimization."""
        try:
            df_work = df.copy()
            
            # Prepare features for Random Forest
            features = ['QTY', 'Line', 'Weight', 'depth', 'height']
            available_features = [f for f in features if f in df_work.columns and df_work[f].sum() != 0]
            
            if len(available_features) < 2:
                return {"success": False, "message": "Insufficient features for Random Forest model"}
            
            # Fill missing values
            for feature in available_features:
                df_work[feature] = df_work[feature].fillna(df_work[feature].median())
            
            X = df_work[available_features].values
            
            # Create target variable (encode current locations numerically)
            if 'Location_Zone' not in self.label_encoders:
                self.label_encoders['Location_Zone'] = LabelEncoder()
                y = self.label_encoders['Location_Zone'].fit_transform(df_work['Location_Zone'].fillna('Unknown'))
            else:
                y = self.label_encoders['Location_Zone'].transform(df_work['Location_Zone'].fillna('Unknown'))
            
            # Train Random Forest
            rf = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=10)
            rf.fit(X, y)
            
            # Generate optimized locations based on pick frequency priority
            df_work['Pick_Priority'] = df_work['Line'].rank(ascending=False, method='dense')
            
            # Create zone hierarchy (front zones for high-priority items)
            unique_zones = sorted(df_work['Location_Zone'].unique())
            zone_priority_map = {zone: idx for idx, zone in enumerate(unique_zones)}
            
            # Sort by pick frequency and assign to better zones
            df_optimized = df_work.sort_values('Line', ascending=False).copy()
            
            # Assign high-frequency items to front zones
            num_zones = len(unique_zones)
            zone_assignments = []
            
            for idx, row in df_optimized.iterrows():
                priority_percentile = row['Pick_Priority'] / len(df_optimized)
                
                if priority_percentile <= 0.2:  # Top 20% get front zones
                    target_zone_idx = min(int(priority_percentile * num_zones * 0.3), num_zones - 1)
                elif priority_percentile <= 0.5:  # Next 30% get middle zones
                    target_zone_idx = min(int(0.3 * num_zones + (priority_percentile - 0.2) * num_zones * 0.4), num_zones - 1)
                else:  # Rest get back zones
                    target_zone_idx = min(int(0.7 * num_zones + (priority_percentile - 0.5) * num_zones * 0.3), num_zones - 1)
                
                target_zone = unique_zones[target_zone_idx]
                zone_assignments.append(target_zone)
            
            df_optimized['New_Location_Zone'] = zone_assignments
            df_optimized['New_Location'] = df_optimized.apply(self._generate_new_location, axis=1)
            
            # Update location columns
            df_result = df_optimized.copy()
            df_result['Location'] = df_result['New_Location']
            df_result['Location_Zone'] = df_result['New_Location_Zone']
            
            # Remove temporary columns
            df_result = df_result.drop(['Pick_Priority', 'New_Location_Zone', 'New_Location'], axis=1, errors='ignore')
            
            return {"success": True, "data": df_result, "message": "Random Forest optimization completed"}
            
        except Exception as e:
            return {"success": False, "message": f"Random Forest error: {str(e)}"}
    
    def _apply_collaborative_filtering(self, df: pd.DataFrame) -> dict:
        """Apply Collaborative Filtering for location optimization."""
        try:
            df_work = df.copy()
            
            # Create item-feature matrix
            features = ['QTY', 'Line', 'Weight', 'depth', 'height']
            available_features = [f for f in features if f in df_work.columns and df_work[f].sum() != 0]
            
            if len(available_features) < 2:
                return {"success": False, "message": "Insufficient features for Collaborative Filtering"}
            
            # Fill missing values and normalize
            for feature in available_features:
                df_work[feature] = df_work[feature].fillna(df_work[feature].median())
            
            # Create feature matrix and calculate similarity
            feature_matrix = df_work[available_features].values
            
            # Normalize features
            feature_matrix = self.scaler.fit_transform(feature_matrix)
            
            # Calculate cosine similarity between products
            similarity_matrix = cosine_similarity(feature_matrix)
            
            # Group similar items together
            df_work['Similarity_Group'] = self._create_similarity_groups(similarity_matrix, df_work)
            
            # Sort by pick frequency within each group
            df_work['Group_Pick_Rank'] = df_work.groupby('Similarity_Group')['Line'].rank(ascending=False)
            
            # Assign locations based on similarity groups and pick frequency
            unique_zones = sorted(df_work['Location_Zone'].unique())
            num_zones = len(unique_zones)
            num_groups = df_work['Similarity_Group'].nunique()
            
            zone_assignments = []
            for idx, row in df_work.iterrows():
                group = row['Similarity_Group']
                pick_rank = row['Group_Pick_Rank']
                total_in_group = df_work[df_work['Similarity_Group'] == group].shape[0]
                
                # Assign zone based on group (similar items together)
                base_zone_idx = int((group / num_groups) * num_zones)
                
                # Fine-tune within group based on pick frequency
                pick_adjustment = min(int((pick_rank - 1) / total_in_group * 0.3 * num_zones), num_zones - 1)
                final_zone_idx = min(base_zone_idx + pick_adjustment, num_zones - 1)
                
                target_zone = unique_zones[final_zone_idx]
                zone_assignments.append(target_zone)
            
            df_work['New_Location_Zone'] = zone_assignments
            df_work['New_Location'] = df_work.apply(self._generate_new_location, axis=1)
            
            # Update location columns
            df_result = df_work.copy()
            df_result['Location'] = df_result['New_Location']
            df_result['Location_Zone'] = df_result['New_Location_Zone']
            
            # Remove temporary columns
            df_result = df_result.drop(['Similarity_Group', 'Group_Pick_Rank', 'New_Location_Zone', 'New_Location'], axis=1, errors='ignore')
            
            return {"success": True, "data": df_result, "message": "Collaborative Filtering optimization completed"}
            
        except Exception as e:
            return {"success": False, "message": f"Collaborative Filtering error: {str(e)}"}
    
    def _apply_ranking_algorithm(self, df: pd.DataFrame) -> dict:
        """Apply Ranking Algorithm for location optimization based on pick frequency."""
        try:
            df_work = df.copy()
            
            # Create comprehensive ranking score
            df_work['Pick_Score'] = df_work['Line'].rank(ascending=False, method='dense')
            
            # Add quantity factor (higher quantity might need more accessible location)
            if df_work['QTY'].sum() > 0:
                df_work['QTY_Score'] = df_work['QTY'].rank(ascending=False, method='dense')
                df_work['Combined_Score'] = (df_work['Pick_Score'] * 0.7) + (df_work['QTY_Score'] * 0.3)
            else:
                df_work['Combined_Score'] = df_work['Pick_Score']
            
            # Sort by combined score (best items first)
            df_sorted = df_work.sort_values('Combined_Score', ascending=True)
            
            # Create zone hierarchy (assign better zones to higher-ranking items)
            unique_zones = sorted(df_work['Location_Zone'].unique())
            num_zones = len(unique_zones)
            
            # Create "front" zones mapping (assuming alphabetically/numerically first zones are better)
            # Zone priority: earlier alphabetically = better accessibility
            zone_priority_map = {zone: idx for idx, zone in enumerate(unique_zones)}
            
            # Assign zones based on ranking
            zone_assignments = []
            total_items = len(df_sorted)
            
            for idx, (_, row) in enumerate(df_sorted.iterrows()):
                # Calculate position percentile
                position_percentile = idx / total_items
                
                # Map percentile to zone (higher priority items get better zones)
                if position_percentile <= 0.15:  # Top 15% get best zones
                    zone_idx = 0
                elif position_percentile <= 0.30:  # Next 15% get second-tier zones
                    zone_idx = min(1, num_zones - 1)
                elif position_percentile <= 0.50:  # Next 20% get mid-tier zones
                    zone_idx = min(int(num_zones * 0.3), num_zones - 1)
                elif position_percentile <= 0.75:  # Next 25% get lower-mid zones
                    zone_idx = min(int(num_zones * 0.6), num_zones - 1)
                else:  # Bottom 25% get back zones
                    zone_idx = min(int(num_zones * 0.8), num_zones - 1)
                
                # Ensure we don't exceed available zones
                zone_idx = min(zone_idx, num_zones - 1)
                target_zone = unique_zones[zone_idx]
                zone_assignments.append(target_zone)
            
            df_sorted['New_Location_Zone'] = zone_assignments
            df_sorted['New_Location'] = df_sorted.apply(self._generate_new_location, axis=1)
            
            # Restore original order by SKU
            df_result = df_sorted.set_index('SKU').loc[df_work['SKU']].reset_index()
            
            # Update location columns
            df_result['Location'] = df_result['New_Location']
            df_result['Location_Zone'] = df_result['New_Location_Zone']
            
            # Remove temporary columns
            df_result = df_result.drop(['Pick_Score', 'QTY_Score', 'Combined_Score', 'New_Location_Zone', 'New_Location'], axis=1, errors='ignore')
            
            return {"success": True, "data": df_result, "message": "Ranking algorithm optimization completed"}
            
        except Exception as e:
            return {"success": False, "message": f"Ranking algorithm error: {str(e)}"}
    
    def _create_similarity_groups(self, similarity_matrix: np.ndarray, df: pd.DataFrame) -> np.ndarray:
        """Create similarity groups based on cosine similarity."""
        try:
            n_items = len(df)
            groups = np.arange(n_items)  # Initialize each item in its own group
            
            # Simple clustering based on similarity threshold
            similarity_threshold = 0.7
            
            for i in range(n_items):
                for j in range(i + 1, n_items):
                    if similarity_matrix[i, j] > similarity_threshold:
                        # Merge groups
                        min_group = min(groups[i], groups[j])
                        max_group = max(groups[i], groups[j])
                        groups[groups == max_group] = min_group
            
            # Renumber groups to be sequential
            unique_groups = np.unique(groups)
            group_mapping = {old_group: new_group for new_group, old_group in enumerate(unique_groups)}
            groups = np.array([group_mapping[group] for group in groups])
            
            return groups
            
        except Exception as e:
            # Fallback to simple grouping by pick frequency quartiles
            pick_quartiles = pd.qcut(df['Line'], q=4, labels=[0, 1, 2, 3], duplicates='drop')
            return pick_quartiles.values
    
    def _generate_new_location(self, row) -> str:
        """Generate new location string based on new zone assignment."""
        try:
            new_zone = row['New_Location_Zone']
            
            # Extract current location parts for reference
            current_location = str(row['Location']).strip()
            location_parts = current_location.split('-')
            
            # Generate new location within the assigned zone
            if len(location_parts) >= 4:
                # Keep the same sub-location pattern but update zone
                zone_parts = new_zone.split('-')
                if len(zone_parts) >= 3:
                    new_location = f"{zone_parts[0]}-{zone_parts[1]}-{zone_parts[2]}-{location_parts[-1]}"
                else:
                    new_location = f"{new_zone}-{location_parts[-1]}"
            else:
                # Simple case: just assign to zone with position 1
                new_location = f"{new_zone}-1"
            
            return new_location
            
        except Exception as e:
            # Fallback: return original location
            return str(row['Location'])
    
    def calculate_optimization_metrics(self, df_original: pd.DataFrame, df_optimized: pd.DataFrame) -> dict:
        """Calculate metrics to show optimization improvement."""
        try:
            metrics = {}
            
            # Calculate pick frequency distribution improvement
            original_zones = df_original.groupby('Location_Zone')['Line'].sum().sort_values(ascending=False)
            optimized_zones = df_optimized.groupby('Location_Zone')['Line'].sum().sort_values(ascending=False)
            
            # Front zone utilization (top 30% of zones by pick frequency)
            top_zones_count = max(1, int(len(original_zones) * 0.3))
            
            original_front_picks = original_zones.head(top_zones_count).sum()
            optimized_front_picks = optimized_zones.head(top_zones_count).sum()
            
            total_picks = df_original['Line'].sum()
            
            metrics['original_front_zone_utilization'] = round((original_front_picks / total_picks) * 100, 2)
            metrics['optimized_front_zone_utilization'] = round((optimized_front_picks / total_picks) * 100, 2)
            metrics['improvement_percentage'] = round(
                ((optimized_front_picks - original_front_picks) / original_front_picks) * 100, 2
            ) if original_front_picks > 0 else 0
            
            # Calculate zone consolidation
            original_zones_used = df_original['Location_Zone'].nunique()
            optimized_zones_used = df_optimized['Location_Zone'].nunique()
            
            metrics['original_zones_used'] = original_zones_used
            metrics['optimized_zones_used'] = optimized_zones_used
            
            # High-frequency item placement improvement
            high_freq_threshold = df_original['Line'].quantile(0.8)
            high_freq_items_original = df_original[df_original['Line'] >= high_freq_threshold]
            high_freq_items_optimized = df_optimized[df_optimized['Line'] >= high_freq_threshold]
            
            # Calculate average "zone priority" (lower index = better zone)
            zone_list = sorted(df_original['Location_Zone'].unique())
            zone_priority_map = {zone: idx for idx, zone in enumerate(zone_list)}
            
            original_avg_priority = high_freq_items_original['Location_Zone'].map(zone_priority_map).mean()
            optimized_avg_priority = high_freq_items_optimized['Location_Zone'].map(zone_priority_map).mean()
            
            metrics['high_frequency_items_count'] = len(high_freq_items_original)
            metrics['original_avg_zone_priority'] = round(original_avg_priority, 2)
            metrics['optimized_avg_zone_priority'] = round(optimized_avg_priority, 2)
            metrics['zone_priority_improvement'] = round(original_avg_priority - optimized_avg_priority, 2)
            
            # Travel distance estimation (simplified)
            # Assume zone index represents distance from picking start point
            original_weighted_distance = sum(
                row['Line'] * zone_priority_map.get(row['Location_Zone'], 0) 
                for _, row in df_original.iterrows()
            )
            optimized_weighted_distance = sum(
                row['Line'] * zone_priority_map.get(row['Location_Zone'], 0) 
                for _, row in df_optimized.iterrows()
            )
            
            metrics['estimated_distance_reduction'] = round(
                ((original_weighted_distance - optimized_weighted_distance) / original_weighted_distance) * 100, 2
            ) if original_weighted_distance > 0 else 0
            
            # Summary metrics
            metrics['total_skus_optimized'] = len(df_optimized)
            metrics['total_moves_required'] = sum(
                1 for i in range(len(df_original)) 
                if df_original.iloc[i]['Location'] != df_optimized.iloc[i]['Location']
            )
            metrics['optimization_coverage'] = round((metrics['total_moves_required'] / len(df_original)) * 100, 2)
            
            return metrics
            
        except Exception as e:
            return {"error": f"Error calculating metrics: {str(e)}"}
    
    def get_model_recommendations(self, df: pd.DataFrame) -> dict:
        """Provide recommendations on which model to use based on data characteristics."""
        try:
            recommendations = {}
            
            data_size = len(df)
            feature_availability = sum([
                1 for col in ['Weight', 'depth', 'height'] 
                if col in df.columns and df[col].sum() > 0
            ])
            pick_variance = df['Line'].var() if 'Line' in df.columns else 0
            zone_diversity = df['Location_Zone'].nunique() if 'Location_Zone' in df.columns else 0
            
            # Ranking algorithm - always recommended for pick frequency optimization
            recommendations['ranking'] = {
                "score": 0.9,
                "reason": "Excellent for optimizing pick frequency and accessibility",
                "best_for": "High pick frequency variance and accessibility optimization"
            }
            
            # Random Forest - good for complex relationships
            rf_score = 0.3 + (feature_availability * 0.2) + (min(data_size / 100, 0.3))
            recommendations['random_forest'] = {
                "score": rf_score,
                "reason": f"Good for complex patterns with {feature_availability} available features",
                "best_for": "Complex product characteristics and multi-factor optimization"
            }
            
            # Collaborative Filtering - good for grouping similar items
            cf_score = 0.4 + (min(zone_diversity / 10, 0.3)) + (min(data_size / 200, 0.2))
            recommendations['collaborative_filtering'] = {
                "score": cf_score,
                "reason": f"Good for grouping similar products across {zone_diversity} zones",
                "best_for": "Similar product grouping and zone consolidation"
            }
            
            # Sort by score
            sorted_recommendations = dict(
                sorted(recommendations.items(), key=lambda x: x[1]['score'], reverse=True)
            )
            
            return {"success": True, "recommendations": sorted_recommendations}
            
        except Exception as e:
            return {"success": False, "message": f"Error generating recommendations: {str(e)}"}