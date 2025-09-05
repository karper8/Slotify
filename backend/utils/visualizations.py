import matplotlib
matplotlib.use('Agg')  # Set backend before importing pyplot
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')
import pandas as pd

class VisualizationGenerator:
    """Fixed visualization generator with memory management."""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        # Setup matplotlib for server environment
        plt.ioff()  # Turn off interactive mode
        self.dpi = 100  # Reduced DPI for memory efficiency
        
    def create_location_heatmap(self, df: pd.DataFrame, title: str = "Location Heatmap") -> dict:
        """Create optimized location heatmap."""
        try:
            self.logger.info(f"Creating heatmap: {title}")
            
            if df.empty:
                return {"success": False, "message": "Empty dataframe provided"}
            
            # Validate required columns
            required_cols = ['Location_Zone', 'QTY', 'Line']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                return {"success": False, "message": f"Missing columns: {missing_cols}"}
            
            # Clean and aggregate data
            df_clean = df.dropna(subset=required_cols)
            if df_clean.empty:
                return {"success": False, "message": "No valid data for visualization"}
            
            # Ensure SKU column exists
            if 'SKU' not in df_clean.columns:
                df_clean = df_clean.copy()
                df_clean['SKU'] = range(len(df_clean))
            
            # Aggregate by zone
            zone_agg = df_clean.groupby('Location_Zone').agg({
                'QTY': 'sum',
                'Line': 'sum',
                'SKU': 'nunique'
            }).reset_index()
            zone_agg.columns = ['Location_Zone', 'Total_QTY', 'Total_Pick_Freq', 'SKU_Count']
            
            # Create visualization
            result = self._create_zone_visualization(zone_agg, title)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in create_location_heatmap: {str(e)}")
            plt.close('all')
            return {"success": False, "message": f"Error creating heatmap: {str(e)}"}
    
    def _create_zone_visualization(self, zone_agg: pd.DataFrame, title: str) -> dict:
        """Create zone visualization with memory management."""
        fig = None
        try:
            # Create figure with appropriate size
            fig_width = 14
            fig_height = max(8, min(len(zone_agg) * 0.3, 16))
            
            fig, axes = plt.subplots(2, 2, figsize=(fig_width, fig_height), dpi=self.dpi)
            
            # Limit display to top 15 zones
            display_limit = 15
            
            # 1. Pick Frequency by Zone
            zone_pick = zone_agg.nlargest(display_limit, 'Total_Pick_Freq')
            if not zone_pick.empty:
                axes[0,0].barh(range(len(zone_pick)), zone_pick['Total_Pick_Freq'], color='skyblue')
                axes[0,0].set_yticks(range(len(zone_pick)))
                axes[0,0].set_yticklabels([str(z)[:20] for z in zone_pick['Location_Zone']], fontsize=8)
                axes[0,0].set_title('Pick Frequency by Zone', fontweight='bold')
                axes[0,0].set_xlabel('Total Picks')
                axes[0,0].grid(axis='x', alpha=0.3)
            
            # 2. Quantity by Zone
            zone_qty = zone_agg.nlargest(display_limit, 'Total_QTY')
            if not zone_qty.empty:
                axes[0,1].barh(range(len(zone_qty)), zone_qty['Total_QTY'], color='lightcoral')
                axes[0,1].set_yticks(range(len(zone_qty)))
                axes[0,1].set_yticklabels([str(z)[:20] for z in zone_qty['Location_Zone']], fontsize=8)
                axes[0,1].set_title('Quantity by Zone', fontweight='bold')
                axes[0,1].set_xlabel('Total Quantity')
                axes[0,1].grid(axis='x', alpha=0.3)
            
            # 3. SKU Count by Zone
            zone_sku = zone_agg.nlargest(display_limit, 'SKU_Count')
            if not zone_sku.empty:
                axes[1,0].barh(range(len(zone_sku)), zone_sku['SKU_Count'], color='lightgreen')
                axes[1,0].set_yticks(range(len(zone_sku)))
                axes[1,0].set_yticklabels([str(z)[:20] for z in zone_sku['Location_Zone']], fontsize=8)
                axes[1,0].set_title('SKU Count by Zone', fontweight='bold')
                axes[1,0].set_xlabel('Number of SKUs')
                axes[1,0].grid(axis='x', alpha=0.3)
            
            # 4. Zone Efficiency
            zone_agg['Efficiency'] = np.where(zone_agg['SKU_Count'] > 0,
                                            zone_agg['Total_Pick_Freq'] / zone_agg['SKU_Count'], 0)
            zone_eff = zone_agg.nlargest(display_limit, 'Efficiency')
            if not zone_eff.empty and zone_eff['Efficiency'].sum() > 0:
                axes[1,1].barh(range(len(zone_eff)), zone_eff['Efficiency'], color='gold')
                axes[1,1].set_yticks(range(len(zone_eff)))
                axes[1,1].set_yticklabels([str(z)[:20] for z in zone_eff['Location_Zone']], fontsize=8)
                axes[1,1].set_title('Zone Efficiency (Picks/SKU)', fontweight='bold')
                axes[1,1].set_xlabel('Picks per SKU')
                axes[1,1].grid(axis='x', alpha=0.3)
            
            plt.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            
            # Convert to base64
            img_base64 = self._fig_to_base64(fig)
            
            return {
                "success": True,
                "image": img_base64,
                "stats": {
                    "total_zones": len(zone_agg),
                    "total_qty": int(zone_agg['Total_QTY'].sum()),
                    "total_pick_freq": int(zone_agg['Total_Pick_Freq'].sum()),
                    "avg_pick_freq_per_zone": round(zone_agg['Total_Pick_Freq'].mean(), 2),
                    "total_skus": int(zone_agg['SKU_Count'].sum())
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error in _create_zone_visualization: {str(e)}")
            raise e
        finally:
            if fig:
                plt.close(fig)
            plt.close('all')
    
    def _fig_to_base64(self, fig) -> str:
        """Convert matplotlib figure to base64 with memory management."""
        buffer = None
        try:
            buffer = BytesIO()
            fig.savefig(buffer, format='png', dpi=self.dpi, bbox_inches='tight',
                       facecolor='white', edgecolor='none', optimize=True)
            buffer.seek(0)
            image_png = buffer.getvalue()
            
            image_base64 = base64.b64encode(image_png).decode('utf-8')
            return f"data:image/png;base64,{image_base64}"
            
        except Exception as e:
            raise Exception(f"Error converting figure to base64: {str(e)}")
        finally:
            if buffer:
                buffer.close()
    
    def create_comprehensive_analytics(self, df: pd.DataFrame) -> dict:
        """Create comprehensive analytics with proper memory management."""
        try:
            if df.empty:
                return {"success": False, "message": "Empty dataframe provided"}
            
            analytics = {}
            
            # Generate summary statistics
            analytics["summary_stats"] = self._generate_summary_statistics(df)
            
            # Create simple distribution chart
            dist_chart = self._create_distribution_chart(df)
            if dist_chart["success"]:
                analytics["distribution"] = dist_chart["image"]
            
            return {"success": True, "analytics": analytics}
            
        except Exception as e:
            self.logger.error(f"Error creating analytics: {str(e)}")
            return {"success": False, "message": f"Error creating analytics: {str(e)}"}
    
    def _create_distribution_chart(self, df: pd.DataFrame) -> dict:
        """Create a simple distribution chart."""
        fig = None
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), dpi=self.dpi)
            
            # Quantity distribution
            if 'QTY' in df.columns and df['QTY'].sum() > 0:
                qty_data = df['QTY'].dropna()
                ax1.hist(qty_data, bins=min(20, len(qty_data.unique())), 
                        color='skyblue', alpha=0.7, edgecolor='black')
                ax1.set_title('Quantity Distribution')
                ax1.set_xlabel('Quantity')
                ax1.set_ylabel('Frequency')
                ax1.grid(True, alpha=0.3)
            
            # Pick frequency distribution
            if 'Line' in df.columns and df['Line'].sum() > 0:
                line_data = df['Line'].dropna()
                ax2.hist(line_data, bins=min(20, len(line_data.unique())), 
                        color='orange', alpha=0.7, edgecolor='black')
                ax2.set_title('Pick Frequency Distribution')
                ax2.set_xlabel('Pick Frequency')
                ax2.set_ylabel('Number of SKUs')
                ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            img_base64 = self._fig_to_base64(fig)
            
            return {"success": True, "image": img_base64}
            
        except Exception as e:
            return {"success": False, "message": f"Error creating distribution chart: {str(e)}"}
        finally:
            if fig:
                plt.close(fig)
            plt.close('all')
    
    def _generate_summary_statistics(self, df: pd.DataFrame) -> dict:
        """Generate summary statistics safely."""
        try:
            stats = {
                "total_skus": len(df),
                "total_quantity": int(df['QTY'].sum()) if 'QTY' in df.columns else 0,
                "total_picks": int(df['Line'].sum()) if 'Line' in df.columns else 0,
                "total_zones": df['Location_Zone'].nunique() if 'Location_Zone' in df.columns else 0,
            }
            
            # Safe calculations with error handling
            try:
                if 'QTY' in df.columns and len(df) > 0:
                    stats["avg_quantity_per_sku"] = round(df['QTY'].mean(), 2)
            except:
                stats["avg_quantity_per_sku"] = 0
            
            try:
                if 'Line' in df.columns and len(df) > 0:
                    stats["avg_picks_per_sku"] = round(df['Line'].mean(), 2)
            except:
                stats["avg_picks_per_sku"] = 0
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error generating statistics: {e}")
            return {"error": f"Error generating statistics: {str(e)}"}