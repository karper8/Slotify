import pandas as pd
import numpy as np
import logging
import os

class DataProcessor:
    """
    Enhanced data processor for warehouse datasets with standardized lowercase columns
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def _normalize_column_names(self, df):
        """
        Normalize column names to lowercase and handle case insensitivity
        """
        # First convert all columns to lowercase
        df.columns = [col.lower().strip() for col in df.columns]
        
        # Standard column mapping to expected format
        column_mapping = {
            # SKU variations
            'product_id': 'sku',
            'item_id': 'sku', 
            'item': 'sku',
            'product': 'sku',
            'product_code': 'sku',
            
            # Location variations
            'loc': 'location',
            'position': 'location',
            'bin': 'location',
            'slot': 'location',
            'warehouse_location': 'location',
            
            # Quantity variations
            'quantity': 'qty',
            'stock': 'qty',
            'inventory': 'qty',
            'stock_qty': 'qty',
            
            # Pick frequency variations (LINE column)
            'picks': 'line',
            'pick_frequency': 'line',
            'frequency': 'line',
            'pick_count': 'line',
            'lines': 'line',
            'orders': 'line',
            'transactions': 'line',
            
            # Weight variations
            'wt': 'weight',
            'wgt': 'weight',
            'mass': 'weight',
            
            # Depth variations
            'dep': 'depth',
            'd': 'depth',
            'length': 'depth',
            
            # Height variations
            'ht': 'height',
            'hgt': 'height',
            'h': 'height',
            
            # Volume variations
            'volume': 'vol',
            'cubic': 'vol',
            'vol_cubic': 'vol',
            'cube': 'vol'
        }
        
        # Apply mappings
        df = df.rename(columns=column_mapping)
        
        return df
    
    def _validate_required_columns(self, df):
        """
        Validate that required columns are present after normalization
        """
        required_columns = ['sku', 'location', 'qty', 'line', 'weight', 'vol']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            # Provide helpful error message with available columns
            available_columns = list(df.columns)
            suggestion_msg = (
                f"Missing required columns: {missing_columns}. "
                f"Available columns: {available_columns}. "
                f"Please ensure your file has columns for: SKU, Location, Quantity, Pick Frequency, Weight, and Volume."
            )
            return False, suggestion_msg
        
        return True, "All required columns present"
    
    def _clean_and_validate_data(self, df):
        """
        Clean and validate the dataset with enhanced error handling
        """
        try:
            initial_rows = len(df)
            self.logger.info(f"Starting data cleaning with {initial_rows} rows")
            
            # Convert columns to appropriate data types with error handling
            numeric_columns = ['qty', 'line', 'weight', 'vol']
            
            # Add optional numeric columns if they exist
            for optional_col in ['depth', 'height']:
                if optional_col in df.columns:
                    numeric_columns.append(optional_col)
            
            # Convert to numeric with proper error handling
            for col in numeric_columns:
                if col in df.columns:
                    original_values = df[col].copy()
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    # Log conversion issues
                    nan_count = df[col].isna().sum() - original_values.isna().sum()
                    if nan_count > 0:
                        self.logger.warning(f"Converted {nan_count} non-numeric values to NaN in column '{col}'")
            
            # Handle missing numeric values intelligently
            for col in ['qty', 'weight', 'vol']:
                if col in df.columns:
                    # For critical columns, fill with appropriate defaults
                    if col == 'qty':
                        df[col] = df[col].fillna(1)  # Default quantity of 1
                    elif col == 'weight':
                        df[col] = df[col].fillna(df[col].median())  # Use median weight
                    elif col == 'vol':
                        df[col] = df[col].fillna(df[col].median())  # Use median volume
            
            # For line (pick frequency), 0 is meaningful (no picks)
            if 'line' in df.columns:
                df['line'] = df['line'].fillna(0)
            
            # Clean string columns
            df['sku'] = df['sku'].astype(str).str.strip()
            df['location'] = df['location'].astype(str).str.strip()
            
            # Remove rows with invalid data
            df = df[df['sku'] != '']  # Remove empty SKUs
            df = df[df['sku'] != 'nan']  # Remove string 'nan' SKUs
            df = df[df['location'] != '']  # Remove empty locations
            df = df[df['location'] != 'nan']  # Remove string 'nan' locations
            df = df[df['qty'] > 0]  # Must have positive quantity
            
            # Handle volume calculation if missing
            if df['vol'].sum() == 0 and 'depth' in df.columns and 'height' in df.columns:
                self.logger.info("Calculating volume from depth, height, and weight")
                df['vol'] = df['depth'] * df['height'] * df['weight']
                # Ensure no zero volumes
                df['vol'] = df['vol'].replace(0, df['vol'].median())
            
            # Remove duplicate SKU-Location combinations (keep highest quantity)
            df = df.sort_values('qty', ascending=False).drop_duplicates(
                subset=['sku', 'location'], keep='first'
            )
            
            final_rows = len(df)
            removed_rows = initial_rows - final_rows
            
            self.logger.info(f"Data cleaning completed: {initial_rows} -> {final_rows} rows ({removed_rows} removed)")
            
            if final_rows == 0:
                return None, "No valid data rows remaining after cleaning"
            
            return df, f"Successfully cleaned data: {removed_rows} invalid rows removed"
            
        except Exception as e:
            self.logger.error(f"Error in data cleaning: {str(e)}")
            raise e
    
    def process_warehouse_data(self, filepath):
        """
        Process warehouse dataset with enhanced validation and cleaning
        """
        try:
            self.logger.info(f"Processing warehouse data from: {filepath}")
            
            # Check file existence
            if not os.path.exists(filepath):
                return {"success": False, "message": f"File not found: {filepath}"}
            
            # Check file size (10MB limit)
            file_size = os.path.getsize(filepath)
            max_size = 10 * 1024 * 1024  # 10MB
            if file_size > max_size:
                return {
                    "success": False, 
                    "message": f"File size ({file_size/1024/1024:.2f}MB) exceeds 10MB limit"
                }
            
            # Load data based on file extension
            file_ext = os.path.splitext(filepath.lower())[1]
            
            try:
                if file_ext == '.csv':
                    # Try different encodings for CSV
                    try:
                        df = pd.read_csv(filepath, encoding='utf-8')
                    except UnicodeDecodeError:
                        try:
                            df = pd.read_csv(filepath, encoding='latin-1')
                        except:
                            df = pd.read_csv(filepath, encoding='cp1252')
                elif file_ext in ['.xlsx', '.xls']:
                    df = pd.read_excel(filepath)
                else:
                    return {
                        "success": False, 
                        "message": f"Unsupported file format: {file_ext}. Please use CSV or Excel files."
                    }
            except Exception as e:
                return {
                    "success": False, 
                    "message": f"Failed to read file: {str(e)}. Please check file format and content."
                }
            
            self.logger.info(f"Loaded data with shape: {df.shape}")
            
            # Check if dataframe is empty
            if df.empty:
                return {"success": False, "message": "The uploaded file is empty"}
            
            # Check minimum data requirements
            if len(df) < 5:
                return {
                    "success": False, 
                    "message": f"Insufficient data: {len(df)} rows found, minimum 5 rows required"
                }
            
            # Normalize column names
            df = self._normalize_column_names(df)
            self.logger.info(f"Normalized columns: {list(df.columns)}")
            
            # Validate required columns
            is_valid, message = self._validate_required_columns(df)
            if not is_valid:
                return {"success": False, "message": message}
            
            # Clean and validate data
            df_cleaned, clean_message = self._clean_and_validate_data(df)
            
            if df_cleaned is None:
                return {"success": False, "message": clean_message}
            
            # Final validation checks
            if len(df_cleaned) < 2:
                return {
                    "success": False, 
                    "message": "Insufficient valid data after cleaning (minimum 2 rows required)"
                }
            
            if df_cleaned['sku'].nunique() < 2:
                return {
                    "success": False, 
                    "message": "Dataset must contain at least 2 unique SKUs"
                }
            
            if df_cleaned['location'].nunique() < 2:
                return {
                    "success": False, 
                    "message": "Dataset must contain at least 2 unique locations"
                }
            
            # Generate summary statistics
            summary_stats = {
                "total_rows": len(df_cleaned),
                "unique_skus": int(df_cleaned['sku'].nunique()),
                "unique_locations": int(df_cleaned['location'].nunique()),
                "total_quantity": float(df_cleaned['qty'].sum()),
                "total_picks": float(df_cleaned['line'].sum()),
                "avg_picks_per_sku": round(df_cleaned.groupby('sku')['line'].sum().mean(), 2),
                "data_quality_score": self._calculate_data_quality_score(df_cleaned)
            }
            
            self.logger.info(f"Processing completed successfully. Summary: {summary_stats}")
            
            return {
                "success": True,
                "message": "Dataset processed successfully",
                "rows": len(df_cleaned),
                "columns": len(df_cleaned.columns),
                "column_names": list(df_cleaned.columns),
                "summary_stats": summary_stats,
                "clean_message": clean_message
            }
            
        except Exception as e:
            self.logger.error(f"Error processing warehouse data: {str(e)}")
            return {"success": False, "message": f"Processing failed: {str(e)}"}
    
    def _calculate_data_quality_score(self, df: pd.DataFrame) -> float:
        """
        Calculate a data quality score (0-100)
        """
        try:
            score = 100.0
            
            # Penalize missing data
            missing_penalty = (df.isna().sum().sum() / (len(df) * len(df.columns))) * 20
            score -= missing_penalty
            
            # Reward data completeness
            if df['line'].sum() > 0:
                score += 10  # Has pick frequency data
            
            if df['vol'].sum() > 0:
                score += 5  # Has volume data
                
            if df['weight'].sum() > 0:
                score += 5  # Has weight data
            
            # Penalize too much duplication
            dup_penalty = (len(df) - df['sku'].nunique()) / len(df) * 10
            score -= dup_penalty
            
            return round(max(0, min(100, score)), 1)
            
        except:
            return 50.0  # Default score if calculation fails
    
    def get_data_preview(self, filepath, n_rows=10):
        """
        Get a preview of the dataset with enhanced error handling
        """
        try:
            # Check file existence and size
            if not os.path.exists(filepath):
                return {"success": False, "message": f"File not found: {filepath}"}
            
            file_size = os.path.getsize(filepath)
            if file_size > 10 * 1024 * 1024:  # 10MB limit
                return {
                    "success": False, 
                    "message": f"File too large: {file_size/1024/1024:.2f}MB (max 10MB)"
                }
            
            # Load data
            file_ext = os.path.splitext(filepath.lower())[1]
            
            try:
                if file_ext == '.csv':
                    # Try different encodings
                    try:
                        df = pd.read_csv(filepath, encoding='utf-8', nrows=n_rows+100)
                    except UnicodeDecodeError:
                        try:
                            df = pd.read_csv(filepath, encoding='latin-1', nrows=n_rows+100)
                        except:
                            df = pd.read_csv(filepath, encoding='cp1252', nrows=n_rows+100)
                elif file_ext in ['.xlsx', '.xls']:
                    df = pd.read_excel(filepath, nrows=n_rows+100)
                else:
                    return {"success": False, "message": f"Unsupported file format: {file_ext}"}
            except Exception as e:
                return {"success": False, "message": f"Failed to read file: {str(e)}"}
            
            # Normalize column names
            df = self._normalize_column_names(df)
            
            # Get preview data
            preview_data = df.head(n_rows).to_dict('records')
            
            # Check column mapping success
            mapped_columns = list(df.columns)
            standard_columns = ['sku', 'location', 'qty', 'line', 'weight', 'vol']
            mapped_standard = [col for col in standard_columns if col in mapped_columns]
            
            return {
                "success": True,
                "preview": preview_data,
                "total_rows": len(df),
                "columns": mapped_columns,
                "mapped_standard_columns": mapped_standard,
                "file_size_mb": round(file_size / 1024 / 1024, 2)
            }
            
        except Exception as e:
            self.logger.error(f"Error getting data preview: {str(e)}")
            return {"success": False, "message": f"Preview failed: {str(e)}"}
    
    def validate_data_quality(self, filepath):
        """
        Enhanced data quality validation with detailed reporting
        """
        try:
            # Load data for validation
            file_ext = os.path.splitext(filepath.lower())[1]
            
            try:
                if file_ext == '.csv':
                    try:
                        df = pd.read_csv(filepath, encoding='utf-8')
                    except UnicodeDecodeError:
                        try:
                            df = pd.read_csv(filepath, encoding='latin-1')
                        except:
                            df = pd.read_csv(filepath, encoding='cp1252')
                elif file_ext in ['.xlsx', '.xls']:
                    df = pd.read_excel(filepath)
                else:
                    return {"success": False, "message": f"Unsupported file format: {file_ext}"}
            except Exception as e:
                return {"success": False, "message": f"Failed to read file: {str(e)}"}
            
            # Normalize columns
            df_original = df.copy()
            df = self._normalize_column_names(df)
            
            quality_report = {
                "total_rows": len(df),
                "total_columns": len(df.columns),
                "original_columns": list(df_original.columns),
                "normalized_columns": list(df.columns),
                "missing_data": {},
                "data_types": {},
                "unique_counts": {},
                "warnings": [],
                "recommendations": [],
                "quality_score": 0
            }
            
            # Analyze each column
            for col in df.columns:
                missing_count = df[col].isnull().sum()
                quality_report["missing_data"][col] = {
                    "count": int(missing_count),
                    "percentage": round((missing_count / len(df)) * 100, 2)
                }
                
                quality_report["data_types"][col] = str(df[col].dtype)
                quality_report["unique_counts"][col] = int(df[col].nunique())
                
                # Generate warnings
                if missing_count > len(df) * 0.5:
                    quality_report["warnings"].append(
                        f"Column '{col}' has {missing_count} missing values "
                        f"({quality_report['missing_data'][col]['percentage']}%)"
                    )
            
            # Check for required columns after normalization
            required_columns = ['sku', 'location', 'qty', 'line', 'weight', 'vol']
            missing_required = [col for col in required_columns if col not in df.columns]
            
            if missing_required:
                quality_report["recommendations"].append(
                    f"Missing required columns after normalization: {missing_required}. "
                    f"Available normalized columns: {list(df.columns)}"
                )
            
            # Check data completeness
            if 'line' in df.columns:
                if df['line'].sum() == 0:
                    quality_report["warnings"].append(
                        "No pick frequency data found. All values in 'line' column are 0 or missing."
                    )
                elif df['line'].isna().sum() > len(df) * 0.8:
                    quality_report["warnings"].append(
                        "Most pick frequency data is missing. This will affect optimization quality."
                    )
            
            if 'qty' in df.columns and df['qty'].sum() == 0:
                quality_report["warnings"].append(
                    "No quantity data found. All values in 'qty' column are 0 or missing."
                )
            
            # Check for duplicates
            if 'sku' in df.columns and 'location' in df.columns:
                duplicate_count = len(df) - len(df.drop_duplicates(subset=['sku', 'location']))
                if duplicate_count > 0:
                    quality_report["warnings"].append(
                        f"Found {duplicate_count} duplicate SKU-Location combinations"
                    )
            
            # Generate recommendations
            if not missing_required and len(quality_report["warnings"]) == 0:
                quality_report["recommendations"].append("Data quality is good for optimization")
            elif len(missing_required) == 0 and len(quality_report["warnings"]) <= 2:
                quality_report["recommendations"].append(
                    "Data quality is acceptable for optimization with minor issues"
                )
            else:
                quality_report["recommendations"].append(
                    "Data quality issues detected. Consider cleaning data before optimization"
                )
            
            # Calculate quality score
            quality_report["quality_score"] = self._calculate_data_quality_score(df)
            
            return {"success": True, "quality_report": quality_report}
            
        except Exception as e:
            self.logger.error(f"Error validating data quality: {str(e)}")
            return {"success": False, "message": f"Data quality validation failed: {str(e)}"}
    
    def prepare_data_for_optimization(self, filepath):
        """
        Prepare data specifically for warehouse slotting optimization
        """
        try:
            # First process the data normally
            process_result = self.process_warehouse_data(filepath)
            if not process_result['success']:
                return process_result
            
            # Load and preprocess the data again for optimization
            file_ext = os.path.splitext(filepath.lower())[1]
            
            if file_ext == '.csv':
                try:
                    df = pd.read_csv(filepath, encoding='utf-8')
                except UnicodeDecodeError:
                    try:
                        df = pd.read_csv(filepath, encoding='latin-1')
                    except:
                        df = pd.read_csv(filepath, encoding='cp1252')
            else:
                df = pd.read_excel(filepath)
            
            # Apply all preprocessing
            df = self._normalize_column_names(df)
            df_cleaned, clean_message = self._clean_and_validate_data(df)
            
            if df_cleaned is None:
                return {"success": False, "message": clean_message}
            
            return {
                "success": True,
                "data": df_cleaned,
                "message": "Data prepared for optimization",
                "optimization_ready": True
            }
            
        except Exception as e:
            self.logger.error(f"Error preparing data for optimization: {str(e)}")
            return {"success": False, "message": f"Data preparation failed: {str(e)}"}