import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Set backend before other imports
import seaborn as sns
import math
import string
import base64
import os
import uuid
from io import BytesIO
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings('ignore')
import logging

class WarehouseSlottingOptimizer:
    """
    Integrated warehouse slotting optimizer with heatmap visualization
    """
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        # Set up logging if not already configured
        if not self.logger.handlers:
            logging.basicConfig(level=logging.INFO)
        
        self.scaler = StandardScaler()
        self.rf_model = None
        self.cluster_to_zone = {}
        self.cache = {}
        
        # Model parameters (hardcoded as requested)
        self.speed = 1.0  # units per second
        self.handling_time = 15  # seconds per pick
        self.kmeans_clusters = 3
        self.dpi = 80  # For image generation
        
        # Setup matplotlib
        plt.ioff()  # Turn off interactive mode
    
    def optimize_slotting(self, df_raw: pd.DataFrame) -> dict:
        """
        Main function to optimize warehouse slotting
        """
        try:
            self.logger.info("Starting warehouse slotting optimization")
            
            # Data preprocessing and validation
            prep_result = self._preprocess_data(df_raw)
            if not prep_result['success']:
                return prep_result
            
            df = prep_result['data']
            self.logger.info(f"Preprocessed data shape: {df.shape}")
            
            # SKU-level aggregation with caching
            sku_agg = self._aggregate_sku_data(df)
            self.logger.info(f"SKU aggregation completed: {len(sku_agg)} SKUs")
            
            # ABC/FSN classification
            sku_classified = self._classify_abc_fsn(sku_agg)
            self.logger.info("ABC/FSN classification completed")
            
            # KMeans clustering and zone assignment
            sku_clustered = self._perform_clustering(sku_classified)
            self.logger.info("KMeans clustering completed")
            
            # Train RandomForest for future predictions
            rf_accuracy = self._train_random_forest(sku_clustered)
            self.logger.info(f"RandomForest trained with accuracy: {rf_accuracy:.4f}")
            
            # Location coordinate parsing and optimization
            optimized_result = self._optimize_locations(sku_clustered, df)
            if not optimized_result['success']:
                return optimized_result
            
            sku_optimized = optimized_result['data']
            self.logger.info("Location optimization completed")
            
            # Calculate performance metrics
            metrics = self._calculate_metrics(sku_optimized)
            
            # Prepare summary
            summary = {
                'total_skus': len(sku_optimized),
                'zones_used': sku_optimized['zone'].nunique(),
                'total_picks': int(sku_optimized['picks'].sum()),
                'estimated_time_saved_sec': round(metrics['total_time_saved'], 1),
                'time_saved_percentage': round(metrics['time_saved_percentage'], 2),
                'rf_accuracy': round(rf_accuracy, 4)
            }
            
            return {
                'success': True,
                'optimized_data': sku_optimized,
                'metrics': metrics,
                'summary': summary,
                'message': 'Warehouse slotting optimization completed successfully'
            }
            
        except Exception as e:
            self.logger.error(f"Error in optimize_slotting: {str(e)}")
            return {'success': False, 'message': f'Optimization failed: {str(e)}'}
    
    def _preprocess_data(self, df_raw: pd.DataFrame) -> dict:
        """
        Preprocess and validate input data - standardized for both models
        """
        try:
            # Data should already be normalized by data_processor
            df = df_raw.copy()
            
            # Validate required columns (lowercase)
            required_columns = ['sku', 'location', 'qty', 'line', 'weight', 'vol']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                return {
                    'success': False, 
                    'message': f'Missing required columns: {missing_columns}. Available: {list(df.columns)}'
                }
            
            # Ensure proper data types
            df['qty'] = pd.to_numeric(df['qty'], errors='coerce').fillna(1)
            df['line'] = pd.to_numeric(df['line'], errors='coerce').fillna(0)
            df['weight'] = pd.to_numeric(df['weight'], errors='coerce')
            df['vol'] = pd.to_numeric(df['vol'], errors='coerce')
            
            # Fill NaN values with medians (avoid issues with empty data)
            if df['weight'].notna().sum() > 0:
                df['weight'] = df['weight'].fillna(df['weight'].median())
            else:
                df['weight'] = df['weight'].fillna(1.0)
                
            if df['vol'].notna().sum() > 0:
                df['vol'] = df['vol'].fillna(df['vol'].median())
            else:
                df['vol'] = df['vol'].fillna(1.0)
            
            # Handle depth and height if available
            if 'depth' in df.columns:
                df['depth'] = pd.to_numeric(df['depth'], errors='coerce').fillna(1.0)
            else:
                df['depth'] = 1.0
                
            if 'height' in df.columns:
                df['height'] = pd.to_numeric(df['height'], errors='coerce').fillna(1.0)
            else:
                df['height'] = 1.0
            
            # Calculate volume if missing or zero
            if df['vol'].sum() == 0:
                df['vol'] = df['depth'] * df['height'] * df['weight']
                self.logger.info("Calculated volume from depth, height, and weight")
            
            # Calculate total volume and weight per row
            df['total_vol_row'] = df['qty'] * df['vol']
            df['total_wt_row'] = df['qty'] * df['weight']
            
            # Clean data
            df = df[df['qty'] > 0]
            df = df[df['sku'].notna() & (df['sku'].astype(str) != '')]
            df = df[df['location'].notna() & (df['location'].astype(str) != '')]
            
            if len(df) == 0:
                return {'success': False, 'message': 'No valid data rows after preprocessing'}
            
            return {'success': True, 'data': df}
            
        except Exception as e:
            return {'success': False, 'message': f'Data preprocessing error: {str(e)}'}
    
    def _aggregate_sku_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate data at SKU level
        """
        try:
            # Cache key for aggregation
            cache_key = f"sku_agg_{len(df)}_{df['sku'].nunique()}"
            
            if cache_key in self.cache:
                self.logger.info("Using cached SKU aggregation")
                return self.cache[cache_key]
            
            sku = df.groupby('sku').agg({
                'qty': 'sum',
                'line': ['count', 'sum'],
                'total_vol_row': 'sum',
                'total_wt_row': 'sum',
                'location': lambda x: x.value_counts().index[0] if len(x.value_counts()) > 0 else x.iloc[0]
            }).reset_index()
            
            # Flatten column names
            sku.columns = ['sku', 'total_qty', 'picks', 'total_line', 'total_volume', 'total_weight', 'most_common_location']
            
            # Use picks (frequency) as the main metric
            sku['avg_qty_per_pick'] = sku['total_qty'] / sku['picks'].replace(0, 1)
            sku['consumption_value'] = sku['total_qty']
            
            # Cache the result
            self.cache[cache_key] = sku
            
            return sku
            
        except Exception as e:
            self.logger.error(f"Error in SKU aggregation: {str(e)}")
            raise e
    
    def _classify_abc_fsn(self, sku: pd.DataFrame) -> pd.DataFrame:
        """
        Perform ABC and FSN classification
        """
        try:
            # ABC Classification
            sku_sorted = sku.sort_values('consumption_value', ascending=False)
            sku_sorted['cumulative_value'] = sku_sorted['consumption_value'].cumsum()
            total_value = sku_sorted['consumption_value'].sum()
            
            if total_value > 0:
                sku_sorted['cumulative_percentage'] = (sku_sorted['cumulative_value'] / total_value) * 100
            else:
                sku_sorted['cumulative_percentage'] = 0
            
            def abc_label(cum_pct):
                if cum_pct <= 80: return 'A'
                if cum_pct <= 95: return 'B'
                return 'C'
            
            sku_sorted['abc'] = sku_sorted['cumulative_percentage'].apply(abc_label)
            
            # FSN Classification
            if len(sku_sorted) > 2:
                p33 = sku_sorted['picks'].quantile(0.33)
                p66 = sku_sorted['picks'].quantile(0.66)
            else:
                p33 = sku_sorted['picks'].min()
                p66 = sku_sorted['picks'].max()
            
            def fsn_label(freq):
                if freq >= p66: return 'F'
                if freq >= p33: return 'S'
                return 'N'
            
            sku_sorted['fsn'] = sku_sorted['picks'].apply(fsn_label)
            
            return sku_sorted.reset_index(drop=True)
            
        except Exception as e:
            self.logger.error(f"Error in ABC/FSN classification: {str(e)}")
            raise e
    
    def _perform_clustering(self, sku: pd.DataFrame) -> pd.DataFrame:
        """
        Perform KMeans clustering for zone assignment
        """
        try:
            # Feature engineering
            sku['log_picks'] = np.log1p(sku['picks'])
            
            features = ['log_picks', 'total_qty', 'avg_qty_per_pick', 'total_volume', 'total_weight']
            X = sku[features].fillna(0)
            
            # Handle case where we have fewer SKUs than clusters
            n_clusters = min(self.kmeans_clusters, len(sku))
            
            if n_clusters <= 1:
                # If only one cluster or less, assign all to zone A
                sku['cluster'] = 0
                sku['zone'] = 'A'
                self.cluster_to_zone = {0: 'A'}
                return sku
            
            # Standardize features
            X_scaled = self.scaler.fit_transform(X)
            
            # KMeans clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            sku['cluster'] = kmeans.fit_predict(X_scaled)
            
            # Map clusters to zones
            cluster_order = sku.groupby('cluster')['picks'].mean().sort_values(ascending=False).index.tolist()
            zone_labels = ['A', 'B', 'C'][:n_clusters]
            self.cluster_to_zone = {cluster_order[i]: zone_labels[i] for i in range(len(cluster_order))}
            sku['zone'] = sku['cluster'].map(self.cluster_to_zone)
            
            return sku
            
        except Exception as e:
            self.logger.error(f"Error in clustering: {str(e)}")
            raise e
    
    def _train_random_forest(self, sku: pd.DataFrame) -> float:
        """
        Train RandomForest model for cluster prediction
        """
        try:
            features = ['log_picks', 'total_qty', 'avg_qty_per_pick', 'total_volume', 'total_weight']
            X = sku[features].fillna(0)
            X_scaled = self.scaler.transform(X)
            y = sku['cluster']
            
            # Check if we have enough data for train/test split
            if len(X) < 10:
                # For small datasets, train on all data
                self.rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
                self.rf_model.fit(X_scaled, y)
                accuracy = self.rf_model.score(X_scaled, y)
            else:
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
                )
                
                # Train RandomForest
                self.rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
                self.rf_model.fit(X_train, y_train)
                
                accuracy = self.rf_model.score(X_test, y_test)
            
            return accuracy
            
        except Exception as e:
            self.logger.error(f"Error training RandomForest: {str(e)}")
            return 0.0
    
    def _optimize_locations(self, sku: pd.DataFrame, df_original: pd.DataFrame) -> dict:
        """
        Optimize location assignments
        """
        try:
            # Parse location coordinates
            location_coords = self._parse_all_locations(df_original)
            
            if not location_coords:
                return {'success': False, 'message': 'Failed to parse location coordinates'}
            
            # Attach coordinates to SKUs
            sku_with_coords = self._attach_coordinates(sku, location_coords)
            
            # Generate optimized locations
            optimized_sku = self._assign_optimal_locations(sku_with_coords, location_coords)
            
            # Calculate time estimates
            optimized_sku = self._calculate_time_estimates(optimized_sku)
            
            return {'success': True, 'data': optimized_sku}
            
        except Exception as e:
            self.logger.error(f"Error in location optimization: {str(e)}")
            return {'success': False, 'message': f'Location optimization failed: {str(e)}'}
    
    def _parse_all_locations(self, df: pd.DataFrame) -> dict:
        """
        Parse all unique locations to coordinates
        """
        try:
            unique_locs = df['location'].astype(str).unique()
            location_coords = {}
            
            for loc in unique_locs:
                x, y = self._parse_location_coordinates(loc)
                location_coords[loc] = {'x': x, 'y': y}
            
            return location_coords
            
        except Exception as e:
            self.logger.error(f"Error parsing locations: {str(e)}")
            return {}
    
    def _parse_location_coordinates(self, loc: str) -> tuple:
        """
        Parse location string to x,y coordinates
        """
        try:
            parts = str(loc).replace('_', '-').replace(' ', '-').split('-')
            first = parts[0] if parts else 'A1'
            
            # Extract letter (default to 'A')
            letter = first[0] if len(first) > 0 and first[0].isalpha() else 'A'
            letter_idx = max(0, ord(letter.upper()) - ord('A'))
            
            # Extract zone number from first part
            zone_num_str = ''.join([c for c in first[1:] if c.isdigit()])
            zone_num = int(zone_num_str) if zone_num_str else 1
            
            # Extract additional numbers from remaining parts
            nums = []
            for part in parts[1:]:
                try:
                    nums.append(int(''.join([c for c in part if c.isdigit()])))
                except:
                    continue
            
            # Calculate x coordinate
            x = float(zone_num)
            if len(nums) > 0:
                x += nums[0] / 100.0
            if len(nums) > 1:
                x += nums[1] / 10000.0
            
            # Calculate y coordinate
            y = float(letter_idx * 10)
            if len(nums) > 2:
                y += nums[2] / 100.0
            
            return x, y
            
        except Exception as e:
            self.logger.warning(f"Error parsing location {loc}: {str(e)}")
            return 0.0, 0.0
    
    def _attach_coordinates(self, sku: pd.DataFrame, location_coords: dict) -> pd.DataFrame:
        """
        Attach coordinate information to SKU data
        """
        try:
            sku_coords = sku.copy()
            
            sku_coords['old_x'] = sku_coords['most_common_location'].map(
                lambda loc: location_coords.get(str(loc), {}).get('x', 0)
            )
            sku_coords['old_y'] = sku_coords['most_common_location'].map(
                lambda loc: location_coords.get(str(loc), {}).get('y', 0)
            )
            
            return sku_coords
            
        except Exception as e:
            self.logger.error(f"Error attaching coordinates: {str(e)}")
            return sku
    
    def _assign_optimal_locations(self, sku: pd.DataFrame, location_coords: dict) -> pd.DataFrame:
        """
        Assign optimal locations based on zones and distance
        """
        try:
            # Convert to DataFrame for easier manipulation
            loc_df = pd.DataFrame([
                {'location': loc, 'x': coords['x'], 'y': coords['y']}
                for loc, coords in location_coords.items()
            ])
            
            if loc_df.empty:
                # Fallback - create dummy locations
                sku['new_location'] = sku['most_common_location']
                sku['new_x'] = sku['old_x']
                sku['new_y'] = sku['old_y']
                return sku
            
            # Calculate distance from start point
            start = np.array([loc_df['x'].min(), loc_df['y'].min()])
            loc_df['dist_to_start'] = np.sqrt(
                (loc_df['x'] - start[0])**2 + (loc_df['y'] - start[1])**2
            )
            
            # Sort locations by distance
            loc_sorted = loc_df.sort_values('dist_to_start').reset_index(drop=True)
            
            # Sort SKUs by zone priority and picks
            zone_priority = {'A': 0, 'B': 1, 'C': 2}
            sku['zone_priority'] = sku['zone'].map(zone_priority).fillna(2)
            sku_sorted = sku.sort_values(
                ['zone_priority', 'picks'], 
                ascending=[True, False]
            ).reset_index(drop=True).copy()
            
            # Assign new locations
            n_skus = len(sku_sorted)
            n_locations = len(loc_sorted)
            
            if n_locations >= n_skus:
                candidate_locs = loc_sorted.iloc[:n_skus]
            else:
                reps = math.ceil(n_skus / n_locations)
                repeated = pd.concat([loc_sorted] * reps, ignore_index=True)
                candidate_locs = repeated.iloc[:n_skus]
            
            sku_sorted['new_location'] = candidate_locs['location'].values[:len(sku_sorted)]
            sku_sorted['new_x'] = candidate_locs['x'].values[:len(sku_sorted)]
            sku_sorted['new_y'] = candidate_locs['y'].values[:len(sku_sorted)]
            
            return sku_sorted
            
        except Exception as e:
            self.logger.error(f"Error in optimal location assignment: {str(e)}")
            raise e
    
    def _calculate_time_estimates(self, sku: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate time estimates for old and new locations
        """
        try:
            # Use min coordinates as start point, with fallback to (0,0)
            old_x_coords = sku['old_x'].dropna()
            old_y_coords = sku['old_y'].dropna()
            
            if len(old_x_coords) > 0 and len(old_y_coords) > 0:
                start = np.array([old_x_coords.min(), old_y_coords.min()])
            else:
                start = np.array([0.0, 0.0])
            
            def estimate_time(picks, x, y, start_point):
                if pd.isna(x) or pd.isna(y):
                    return picks * self.handling_time
                dist = math.hypot(float(x) - start_point[0], float(y) - start_point[1])
                per_pick_time = dist / self.speed + self.handling_time
                return picks * per_pick_time
            
            sku['old_time_sec'] = sku.apply(
                lambda row: estimate_time(row['picks'], row['old_x'], row['old_y'], start), 
                axis=1
            )
            sku['new_time_sec'] = sku.apply(
                lambda row: estimate_time(row['picks'], row['new_x'], row['new_y'], start), 
                axis=1
            )
            sku['time_saved_sec'] = sku['old_time_sec'] - sku['new_time_sec']
            
            return sku
            
        except Exception as e:
            self.logger.error(f"Error calculating time estimates: {str(e)}")
            raise e
    
    def _calculate_metrics(self, sku: pd.DataFrame) -> dict:
        """
        Calculate optimization metrics
        """
        try:
            total_old_time = sku['old_time_sec'].sum()
            total_new_time = sku['new_time_sec'].sum()
            total_time_saved = total_old_time - total_new_time
            time_saved_percentage = (total_time_saved / total_old_time * 100) if total_old_time > 0 else 0
            
            metrics = {
                'total_old_time_sec': round(total_old_time, 1),
                'total_new_time_sec': round(total_new_time, 1),
                'total_time_saved': round(total_time_saved, 1),
                'time_saved_percentage': round(time_saved_percentage, 2),
                'total_skus': len(sku),
                'moves_required': sum(1 for _, row in sku.iterrows() 
                                    if str(row['most_common_location']) != str(row['new_location'])),
                'zone_distribution': sku['zone'].value_counts().to_dict()
            }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating metrics: {str(e)}")
            return {}
    
    def generate_heatmaps_files(self, sku_optimized: pd.DataFrame, output_folder: str, basename: str) -> dict:
        """
        Generate heatmaps and save as files, return filenames
        """
        try:
            self.logger.info("Generating heatmap files")
            
            # Ensure output folder exists
            os.makedirs(output_folder, exist_ok=True)
            
            # Generate before heatmap using actual current locations
            before_filename = f"{basename}_before.png"
            before_path = os.path.join(output_folder, before_filename)
            before_success = self._create_and_save_heatmap(
                sku_optimized, 'old_x', 'old_y', 'Before Slotting Optimization', before_path, 'before'
            )
            
            # Generate after heatmap using optimized new locations
            after_filename = f"{basename}_after.png"
            after_path = os.path.join(output_folder, after_filename)
            after_success = self._create_and_save_heatmap(
                sku_optimized, 'new_x', 'new_y', 'After Slotting Optimization', after_path, 'after'
            )
            
            if not before_success or not after_success:
                # Graceful fallback - try simple scatter plots
                self.logger.warning("Standard heatmaps failed, trying simple fallback")
                before_success = self._create_simple_scatter_file(
                    sku_optimized, 'old_x', 'old_y', 'Before Optimization', before_path
                )
                after_success = self._create_simple_scatter_file(
                    sku_optimized, 'new_x', 'new_y', 'After Optimization', after_path
                )
            
            if not before_success or not after_success:
                return {'success': False, 'message': 'Failed to generate heatmap files'}
            
            return {
                'success': True,
                'before_filename': before_filename,
                'after_filename': after_filename,
                'message': 'Heatmap files generated successfully'
            }
            
        except Exception as e:
            self.logger.error(f"Error generating heatmap files: {str(e)}")
            return {'success': False, 'message': f'Heatmap file generation failed: {str(e)}'}
    
    def _build_heatmap_matrix(self, sku: pd.DataFrame, x_col: str, y_col: str, heatmap_type: str = 'before') -> dict:
        """
        Build matrix for warehouse-style heatmap visualization based on actual location data
        """
        try:
            tmp = sku[[x_col, y_col, 'picks']].dropna().copy()
            
            if len(tmp) == 0:
                return {'success': False, 'message': 'No valid coordinate data'}
            
            # Get coordinate ranges from actual data
            x_vals = tmp[x_col].values
            y_vals = tmp[y_col].values
            
            # Determine warehouse layout based on actual coordinate distribution
            x_min, x_max = np.min(x_vals), np.max(x_vals)
            y_min, y_max = np.min(y_vals), np.max(y_vals)
            
            # Create bins that reflect actual location distribution
            # For columns: use actual x-coordinate range
            n_x_bins = min(max(20, int(x_max - x_min) + 5), 50)
            x_edges = np.linspace(x_min - 0.5, x_max + 0.5, n_x_bins + 1)
            
            # For rows: map y-coordinates to zones and positions within zones
            # Group y-coordinates into natural clusters (zones)
            unique_y = sorted(tmp[y_col].unique())
            
            # Create zone structure based on y-coordinate distribution
            if len(unique_y) <= 3:
                # Simple case: few distinct y values
                zone_breaks = unique_y
            else:
                # More complex: create zones based on y-coordinate clustering
                y_sorted = np.sort(unique_y)
                # Look for natural breaks in y-coordinates
                y_diffs = np.diff(y_sorted)
                if len(y_diffs) > 0:
                    large_gaps = y_diffs > np.percentile(y_diffs, 75)
                    zone_breaks = [y_sorted[0]]
                    for i, has_gap in enumerate(large_gaps):
                        if has_gap:
                            zone_breaks.append(y_sorted[i + 1])
                    if zone_breaks[-1] != y_sorted[-1]:
                        zone_breaks.append(y_sorted[-1])
                else:
                    zone_breaks = unique_y
            
            # Create warehouse grid with zones and aisles
            zone_letters = list(string.ascii_uppercase[:min(10, len(zone_breaks))])
            ylabels = []
            y_to_matrix_row = {}
            
            matrix_row = 0
            for zone_idx, zone_letter in enumerate(zone_letters):
                if zone_idx < len(zone_breaks):
                    # Add 2 rows per zone (like A1, A2)
                    for sub_row in [1, 2]:
                        row_label = f"{zone_letter}{sub_row}"
                        ylabels.append(row_label)
                        
                        # Map y-coordinates in this zone to this matrix row
                        if zone_idx < len(zone_breaks) - 1:
                            y_range = (zone_breaks[zone_idx], zone_breaks[zone_idx + 1])
                        else:
                            y_range = (zone_breaks[zone_idx], y_max + 1)
                        
                        # Assign y-coordinates to this matrix row
                        mask = (tmp[y_col] >= y_range[0]) & (tmp[y_col] < y_range[1])
                        for idx in tmp[mask].index:
                            y_to_matrix_row[idx] = matrix_row
                        
                        matrix_row += 1
                
                # Add aisle row between zones
                if zone_idx < len(zone_letters) - 1:
                    ylabels.append("AISLE")
                    matrix_row += 1
            
            # Initialize heatmap matrix
            n_rows = len(ylabels)
            n_cols = len(x_edges) - 1
            H = np.full((n_rows, n_cols), np.nan)
            
            # Map actual coordinates to matrix positions and aggregate picks
            coord_to_picks = {}
            
            for idx, row in tmp.iterrows():
                x_val = row[x_col]
                y_val = row[y_col]
                picks = row['picks']
                
                # Find which x-bin this coordinate belongs to
                col_idx = np.digitize(x_val, x_edges) - 1
                col_idx = max(0, min(col_idx, n_cols - 1))
                
                # Find which matrix row this y-coordinate maps to
                matrix_row = y_to_matrix_row.get(idx, 0)
                
                # Skip aisle rows for data placement
                if matrix_row < len(ylabels) and ylabels[matrix_row] != "AISLE":
                    pos_key = (matrix_row, col_idx)
                    if pos_key not in coord_to_picks:
                        coord_to_picks[pos_key] = 0
                    coord_to_picks[pos_key] += picks
            
            # Fill the heatmap matrix with aggregated pick data
            for (matrix_row, col_idx), total_picks in coord_to_picks.items():
                if 0 <= matrix_row < n_rows and 0 <= col_idx < n_cols:
                    H[matrix_row][col_idx] = total_picks
            
            # Create column labels based on actual x-coordinate bins
            xlabels = []
            for i in range(n_cols):
                bin_center = (x_edges[i] + x_edges[i + 1]) / 2
                xlabels.append(f"{bin_center:.1f}")
            
            # If no data was placed, add some fallback data
            if np.all(np.isnan(H)):
                # Place data in first few non-aisle rows
                data_placed = 0
                for row_idx in range(min(n_rows, len(tmp))):
                    if row_idx < len(ylabels) and ylabels[row_idx] != "AISLE" and data_placed < len(tmp):
                        col_idx = data_placed % n_cols
                        H[row_idx][col_idx] = tmp.iloc[data_placed]['picks']
                        data_placed += 1
            
            return {
                'success': True,
                'matrix': H,
                'ylabels': ylabels,
                'xlabels': xlabels
            }
            
        except Exception as e:
            self.logger.error(f"Error building heatmap matrix: {str(e)}")
            return {'success': False, 'message': f'Matrix building failed: {str(e)}'}

    def _create_and_save_heatmap(self, sku: pd.DataFrame, x_col: str, y_col: str, title: str, filepath: str, heatmap_type: str = 'before') -> bool:
        """
        Create and save warehouse-style heatmap to file based on actual location data
        """
        fig = None
        try:
            # Build heatmap matrix using actual coordinate data
            matrix_result = self._build_heatmap_matrix(sku, x_col, y_col, heatmap_type)
            if not matrix_result['success']:
                return False
            
            H = matrix_result['matrix']
            ylabels = matrix_result['ylabels']
            xlabels = matrix_result['xlabels']
            
            # Create visualization with warehouse-appropriate dimensions
            fig_width = max(12, len(xlabels) * 0.25)
            fig_height = max(8, len(ylabels) * 0.3)
            fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=self.dpi)
            
            # Create mask for NaN values
            mask = np.isnan(H)
            
            # Set color range based on actual data
            if not np.all(mask):
                vmin = 0  # Always start from 0 for better visualization
                vmax = np.nanmax(H)
                if vmax == vmin:
                    vmax = vmin + 1
            else:
                vmin, vmax = 0, 1
            
            # Create custom colormap matching warehouse heatmap style
            colors = ['#FFFFFF', '#FFEEEE', '#FFDDDD', '#FFCCCC', '#FFAAAA', 
                    '#FF8888', '#FF6666', '#FF4444', '#FF2222', '#CC0000', '#990000']
            custom_cmap = LinearSegmentedColormap.from_list("warehouse", colors)
            
            # Create the heatmap using actual location-based data
            im = ax.imshow(H, cmap=custom_cmap, vmin=vmin, vmax=vmax, aspect='auto')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
            cbar.set_label('Pick Frequency', rotation=270, labelpad=20, fontsize=12)
            
            # Set ticks and labels based on actual coordinate mapping
            ax.set_xticks(range(0, len(xlabels), max(1, len(xlabels)//20)))  # Show every nth label to avoid crowding
            ax.set_yticks(range(len(ylabels)))
            ax.set_xticklabels([xlabels[i] for i in range(0, len(xlabels), max(1, len(xlabels)//20))], fontsize=8)
            ax.set_yticklabels(ylabels, fontsize=8)
            
            # Style the plot to match warehouse format
            ax.set_xlabel('Location X-Coordinate', fontsize=12, fontweight='bold')
            ax.set_ylabel('Warehouse Zone/Row', fontsize=12, fontweight='bold')
            
            # Add descriptive title based on heatmap type
            if heatmap_type == 'before':
                subtitle = f'Current SKU Distribution (Total Picks: {int(sku["picks"].sum())})'
            else:
                subtitle = f'Optimized SKU Distribution (Total Picks: {int(sku["picks"].sum())})'
            
            ax.set_title(f'{title} - Level 0\n{subtitle}', fontsize=14, fontweight='bold', pad=20)
            
            # Add grid lines for better visibility
            ax.set_xticks(np.arange(-0.5, len(xlabels), 1), minor=True)
            ax.set_yticks(np.arange(-0.5, len(ylabels), 1), minor=True)
            ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
            
            # Highlight aisle rows with different background
            for i, label in enumerate(ylabels):
                if label == "AISLE":
                    ax.axhspan(i-0.5, i+0.5, facecolor='lightgray', alpha=0.4, zorder=0)
            
            # Rotate x-axis labels for better readability if many columns
            if len(xlabels) > 15:
                ax.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            
            # Save to file
            plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error creating warehouse heatmap: {str(e)}")
            return False
        finally:
            if fig:
                plt.close(fig)
            plt.close('all')

    def ensure_consistent_heatmap_data(self, result_df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure the result dataframe has all necessary columns for consistent heatmap generation
        This method should be called by both Random Forest and Ranking models
        """
        # Ensure required columns exist for heatmap generation
        required_cols = ['sku', 'picks', 'old_x', 'old_y', 'new_x', 'new_y']
        
        for col in required_cols:
            if col not in result_df.columns:
                if col in ['old_x', 'new_x']:
                    result_df[col] = np.random.uniform(0, 20, len(result_df))
                elif col in ['old_y', 'new_y']:
                    result_df[col] = np.random.uniform(0, 10, len(result_df))
                elif col == 'picks':
                    result_df[col] = 1
                elif col == 'sku':
                    result_df[col] = [f'SKU_{i}' for i in range(len(result_df))]
        
        return result_df
    
    def _create_simple_scatter_file(self, sku: pd.DataFrame, x_col: str, y_col: str, title: str, filepath: str) -> bool:
        """
        Create simple scatter plot fallback and save to file
        """
        fig = None
        try:
            clean_data = sku[[x_col, y_col, 'picks']].dropna()
            if len(clean_data) == 0:
                return False
            
            fig, ax = plt.subplots(figsize=(10, 6), dpi=self.dpi)
            
            scatter = ax.scatter(
                clean_data[x_col], 
                clean_data[y_col], 
                c=clean_data['picks'],
                cmap='YlOrRd',
                s=50,
                alpha=0.7
            )
            
            plt.colorbar(scatter, ax=ax, label='Picks')
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_xlabel('X Position', fontsize=10)
            ax.set_ylabel('Y Position', fontsize=10)
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight', facecolor='white')
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error creating scatter plot: {str(e)}")
            return False
        finally:
            if fig:
                plt.close(fig)
            plt.close('all')
    
    def predict_new_sku_zone(self, sku_features: dict) -> dict:
        """
        Predict zone for a new SKU using trained RandomForest
        
        Args:
            sku_features: dict with keys ['picks', 'total_qty', 'total_volume', 'total_weight']
        
        Returns:
            dict with prediction results
        """
        try:
            if self.rf_model is None:
                return {'success': False, 'message': 'Model not trained yet'}
            
            # Prepare features
            log_picks = np.log1p(sku_features.get('picks', 1))
            total_qty = sku_features.get('total_qty', 1)
            avg_qty_per_pick = total_qty / max(1, sku_features.get('picks', 1))
            total_volume = sku_features.get('total_volume', 1)
            total_weight = sku_features.get('total_weight', 1)
            
            features = np.array([[log_picks, total_qty, avg_qty_per_pick, total_volume, total_weight]])
            features_scaled = self.scaler.transform(features)
            
            # Predict
            cluster_pred = self.rf_model.predict(features_scaled)[0]
            zone_pred = self.cluster_to_zone.get(cluster_pred, 'C')
            
            # Get prediction probabilities
            probabilities = self.rf_model.predict_proba(features_scaled)[0]
            
            return {
                'success': True,
                'predicted_zone': zone_pred,
                'predicted_cluster': int(cluster_pred),
                'confidence': float(np.max(probabilities)),
                'all_probabilities': {f'cluster_{i}': float(prob) for i, prob in enumerate(probabilities)}
            }
            
        except Exception as e:
            self.logger.error(f"Error predicting SKU zone: {str(e)}")
            return {'success': False, 'message': f'Prediction failed: {str(e)}'}


# Example usage and testing functions
def example_usage():
    """
    Example of how to use the WarehouseSlottingOptimizer
    """
    # Create sample data
    sample_data = pd.DataFrame({
        'sku': ['SKU001', 'SKU002', 'SKU003', 'SKU004', 'SKU005'] * 20,
        'location': ['A1-01-01', 'B2-02-01', 'C3-03-01', 'A1-01-02', 'B2-02-02'] * 20,
        'qty': np.random.randint(1, 100, 100),
        'line': np.random.randint(1, 50, 100),
        'weight': np.random.uniform(0.1, 10.0, 100),
        'vol': np.random.uniform(0.01, 1.0, 100)
    })
    
    # Initialize optimizer
    optimizer = WarehouseSlottingOptimizer()
    
    # Run optimization
    result = optimizer.optimize_slotting(sample_data)
    
    if result['success']:
        print("Optimization successful!")
        print(f"Summary: {result['summary']}")
        
        # Generate heatmaps
        heatmap_result = optimizer.generate_heatmaps_files(
            result['optimized_data'], 
            './output', 
            'warehouse_optimization'
        )
        
        if heatmap_result['success']:
            print(f"Heatmaps saved: {heatmap_result['before_filename']}, {heatmap_result['after_filename']}")
        
        # Test prediction for new SKU
        new_sku_features = {
            'picks': 25,
            'total_qty': 500,
            'total_volume': 50.0,
            'total_weight': 25.0
        }
        
        prediction = optimizer.predict_new_sku_zone(new_sku_features)
        if prediction['success']:
            print(f"New SKU predicted zone: {prediction['predicted_zone']} (confidence: {prediction['confidence']:.3f})")
    else:
        print(f"Optimization failed: {result['message']}")


if __name__ == "__main__":
    example_usage()