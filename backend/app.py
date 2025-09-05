from flask import Flask, request, jsonify, session, send_from_directory, send_file, after_this_request
from flask_cors import CORS
import pandas as pd
import os
import traceback
import logging
from werkzeug.utils import secure_filename
import uuid
from datetime import datetime, timedelta
import json
import tempfile

# Import utility modules
<<<<<<< HEAD
from backend.utils.data_processor import DataProcessor
from backend.utils.auth import AuthManager
from backend.warehouse_slotting_optimizer import WarehouseSlottingOptimizer
from backend.models.ranking_slotting_model import RankingSlottingModel
=======
from utils.data_processor import DataProcessor
from utils.auth import AuthManager
from warehouse_slotting_optimizer import WarehouseSlottingOptimizer
from models.ranking_slotting_model import RankingSlottingModel
>>>>>>> e4d8337bf2566ee62f3d8023f356caa52a09958e

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-change-in-production'
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB max file size
app.config['UPLOAD_FOLDER'] = 'temp_uploads'
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=24)
app.config['HEATMAP_FOLDER'] = 'static/heatmaps'
app.config['RESULTS_FOLDER'] = 'static/results'

# Configure for larger responses
app.config['MAX_COOKIE_SIZE'] = 4093
app.config['JSON_SORT_KEYS'] = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
app.logger.setLevel(logging.INFO)

# Enable CORS for frontend with larger response support
CORS(app, supports_credentials=True, max_age=3600)

# Initialize utility classes
data_processor = DataProcessor()
auth_manager = AuthManager()
slotting_optimizer = WarehouseSlottingOptimizer()
ranking_model = RankingSlottingModel()

# Create required directories
for folder in [app.config['UPLOAD_FOLDER'], app.config['HEATMAP_FOLDER'], app.config['RESULTS_FOLDER']]:
    os.makedirs(folder, exist_ok=True)

# Error handlers
@app.errorhandler(413)
def file_too_large(error):
    return jsonify({"error": "File size exceeds 10MB limit", "code": 413}), 413

@app.errorhandler(500)
def internal_server_error(error):
    app.logger.error(f"Internal server error: {str(error)}")
    return jsonify({"error": "Internal server error occurred", "code": 500}), 500

# Utility functions
def cleanup_user_files():
    """Clean up user's session files"""
    try:
        if 'dataset_file' in session:
            try:
                if os.path.exists(session['dataset_file']):
                    os.remove(session['dataset_file'])
                    app.logger.info(f"Cleaned up dataset file: {session['dataset_file']}")
            except Exception:
                pass
        
        # Clear results from session
        session.pop('optimization_results', None)
        session.pop('ranking_results', None)
        session.pop('dataset_info', None)
    except Exception as e:
        app.logger.error(f"Error cleaning up user files: {str(e)}")

def load_stored_results(results_id: str) -> dict:
    """Load results from disk by ID"""
    try:
        results_path = os.path.join(app.config['RESULTS_FOLDER'], f"{results_id}.json")
        if not os.path.exists(results_path):
            return None
        
        with open(results_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        app.logger.error(f"Error loading stored results: {str(e)}")
        return None

def save_results_to_disk(results_payload: dict) -> str:
    """Save results to disk and return results ID"""
    try:
        results_id = str(uuid.uuid4())
        results_path = os.path.join(app.config['RESULTS_FOLDER'], f"{results_id}.json")
        
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results_payload, f, indent=2, default=str)
        
        return results_id
    except Exception as e:
        app.logger.error(f"Error saving results to disk: {str(e)}")
        raise e

# Authentication endpoints
@app.route('/api/signup', methods=['POST'])
def signup():
    try:
        data = request.get_json()
        
        if not data or not data.get('username') or not data.get('password'):
            return jsonify({"error": "Username and password are required", "code": 400}), 400
        
        result = auth_manager.create_user(data['username'], data['password'])
        
        if result['success']:
            session['user_id'] = result['user_id']
            session['username'] = data['username']
            session.permanent = True
            return jsonify({"message": "User created successfully", "username": data['username']})
        else:
            return jsonify({"error": result['message'], "code": 400}), 400
            
    except Exception as e:
        app.logger.error(f"Signup error: {str(e)}")
        return jsonify({"error": "Signup failed", "code": 500}), 500

@app.route('/api/login', methods=['POST'])
def login():
    try:
        data = request.get_json()
        
        if not data or not data.get('username') or not data.get('password'):
            return jsonify({"error": "Username and password are required", "code": 400}), 400
        
        result = auth_manager.authenticate_user(data['username'], data['password'])
        
        if result['success']:
            session['user_id'] = result['user_id']
            session['username'] = data['username']
            session.permanent = True
            return jsonify({"message": "Login successful", "username": data['username']})
        else:
            return jsonify({"error": "Invalid credentials", "code": 401}), 401
            
    except Exception as e:
        app.logger.error(f"Login error: {str(e)}")
        return jsonify({"error": "Login failed", "code": 500}), 500

@app.route('/api/logout', methods=['POST'])
def logout():
    try:
        cleanup_user_files()
        session.clear()
        return jsonify({"message": "Logout successful"})
    except Exception as e:
        app.logger.error(f"Logout error: {str(e)}")
        return jsonify({"error": "Logout failed", "code": 500}), 500

@app.route('/api/session', methods=['GET'])
def check_session():
    try:
        if 'user_id' in session and 'username' in session:
            return jsonify({
                "authenticated": True, 
                "username": session['username'],
                "has_dataset": 'dataset_file' in session,
                "has_rf_results": 'optimization_results' in session,
                "has_ranking_results": 'ranking_results' in session
            })
        else:
            return jsonify({"authenticated": False})
    except Exception as e:
        app.logger.error(f"Session check error: {str(e)}")
        return jsonify({"authenticated": False})

# Dataset management endpoints
@app.route('/api/upload-dataset', methods=['POST'])
def upload_dataset():
    try:
        if 'user_id' not in session:
            return jsonify({"error": "Authentication required", "code": 401}), 401
        
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded", "code": 400}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({"error": "No file selected", "code": 400}), 400
        
        allowed_extensions = ['.csv', '.xlsx', '.xls']
        file_ext = os.path.splitext(file.filename.lower())[1]
        if file_ext not in allowed_extensions:
            return jsonify({"error": "Only CSV and Excel files are allowed", "code": 400}), 400
        
        cleanup_user_files()
        
        filename = secure_filename(f"{uuid.uuid4()}_{file.filename}")
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        file.save(filepath)
        app.logger.info(f"File saved to: {filepath}")
        
        result = data_processor.process_warehouse_data(filepath)
        
        if not result['success']:
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({"error": result['message'], "code": 400}), 400
        
        session['dataset_file'] = filepath
        session['dataset_info'] = {
            'filename': file.filename,
            'rows': result['rows'],
            'columns': result['columns'],
            'column_names': result['column_names'],
            'upload_time': datetime.now().isoformat(),
            'summary_stats': result['summary_stats']
        }
        
        app.logger.info(f"Dataset uploaded successfully: {filepath}")
        
        return jsonify({
            "message": "Dataset uploaded and validated successfully",
            "info": session['dataset_info']
        })
        
    except Exception as e:
        app.logger.error(f"Upload error: {str(e)}\n{traceback.format_exc()}")
        if 'filepath' in locals() and os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({"error": "Upload failed", "code": 500}), 500

@app.route('/api/dataset-preview', methods=['GET'])
def get_dataset_preview():
    try:
        if 'user_id' not in session:
            return jsonify({"error": "Authentication required", "code": 401}), 401
        
        if 'dataset_file' not in session:
            return jsonify({"error": "No dataset found. Please upload your dataset.", "code": 400}), 400
        
        dataset_filepath = session['dataset_file']
        
        if not os.path.exists(dataset_filepath):
            return jsonify({"error": "Dataset file not found. Please re-upload.", "code": 400}), 400
        
        preview_result = data_processor.get_data_preview(dataset_filepath, n_rows=20)
        
        if not preview_result['success']:
            return jsonify({"error": preview_result['message'], "code": 400}), 400
        
        return jsonify({
            "message": "Dataset preview generated successfully",
            "preview": preview_result
        })
        
    except Exception as e:
        app.logger.error(f"Dataset preview error: {str(e)}")
        return jsonify({"error": "Failed to generate preview", "code": 500}), 500

# Random Forest Model endpoint
@app.route('/api/warehouse-slotting', methods=['POST'])
def warehouse_slotting_optimization():
    try:
        if 'user_id' not in session:
            return jsonify({"error": "Authentication required", "code": 401}), 401
        
        if 'dataset_file' not in session:
            return jsonify({"error": "No dataset found. Please upload your dataset.", "code": 400}), 400
        
        dataset_filepath = session['dataset_file']
        
        if not os.path.exists(dataset_filepath):
            return jsonify({"error": "Dataset file not found. Please re-upload.", "code": 400}), 400
        
        app.logger.info(f"=== Starting Random Forest optimization ===")
        
        prep_result = data_processor.prepare_data_for_optimization(dataset_filepath)
        if not prep_result['success']:
            return jsonify({"error": prep_result['message'], "code": 400}), 400
        
        df_prepared = prep_result['data']
        app.logger.info(f"Data prepared successfully with shape: {df_prepared.shape}")
        
        optimization_result = slotting_optimizer.optimize_slotting(df_prepared)
        
        if not optimization_result['success']:
            return jsonify({"error": optimization_result['message'], "code": 400}), 400
        
        heatmap_files = slotting_optimizer.generate_heatmaps_files(
            optimization_result['optimized_data'],
            app.config['HEATMAP_FOLDER'],
            basename=f"rf_{uuid.uuid4()}"
        )
        
        if not heatmap_files['success']:
            app.logger.warning(f"Heatmap generation failed: {heatmap_files['message']}")
            heatmap_files = {'before_filename': None, 'after_filename': None}
        
        try:
            detailed_summary = slotting_optimizer.get_optimization_summary(optimization_result['optimized_data'])
        except AttributeError:
            detailed_summary = {
                'total_skus': len(optimization_result['optimized_data']),
                'zones_used': optimization_result['optimized_data']['zone'].nunique(),
                'total_picks': int(optimization_result['optimized_data']['picks'].sum())
            }
        
        results_payload = {
            'timestamp': datetime.now().isoformat(),
            'metrics': optimization_result['metrics'],
            'summary': optimization_result['summary'],
            'detailed_summary': detailed_summary,
            'optimized_data': optimization_result['optimized_data'].to_dict('records'),
            'model': 'random_forest'
        }
        
        results_id = save_results_to_disk(results_payload)
        
        session['optimization_results'] = {
            'id': results_id,
            'timestamp': results_payload['timestamp'],
            'model': 'random_forest'
        }
        
        app.logger.info("=== Random Forest optimization completed ===")
        
        response_data = {
            "message": "Random Forest optimization completed successfully",
            "optimization_metrics": optimization_result['metrics'],
            "optimization_summary": optimization_result['summary'],
            "detailed_summary": detailed_summary,
            "results_id": results_id,
            "model": "random_forest",
            "data_stats": {
                "original_rows": len(df_prepared),
                "optimized_skus": len(optimization_result['optimized_data']),
                "total_picks": int(optimization_result['optimized_data']['picks'].sum()),
                "zones_used": optimization_result['optimized_data']['zone'].nunique()
            }
        }
        
        if heatmap_files.get('before_filename') and heatmap_files.get('after_filename'):
            response_data.update({
                "before_heatmap_url": f"/api/heatmaps/{heatmap_files['before_filename']}",
                "after_heatmap_url": f"/api/heatmaps/{heatmap_files['after_filename']}"
            })
        
        return jsonify(response_data)
        
    except Exception as e:
        app.logger.error(f"Random Forest optimization error: {str(e)}\n{traceback.format_exc()}")
        return jsonify({"error": "Random Forest optimization failed", "code": 500}), 500

# Ranking model endpoint
@app.route('/api/ranking-slotting', methods=['POST'])
def ranking_slotting_optimization():
    try:
        if 'user_id' not in session:
            return jsonify({"error": "Authentication required", "code": 401}), 401
        
        if 'dataset_file' not in session:
            return jsonify({"error": "No dataset found. Please upload your dataset.", "code": 400}), 400
        
        dataset_filepath = session['dataset_file']
        
        if not os.path.exists(dataset_filepath):
            return jsonify({"error": "Dataset file not found. Please re-upload.", "code": 400}), 400

        app.logger.info(f"=== Starting Ranking optimization ===")

        prep_result = data_processor.prepare_data_for_optimization(dataset_filepath)
        if not prep_result['success']:
            return jsonify({"error": prep_result['message'], "code": 400}), 400

        df_prepared = prep_result['data']
        app.logger.info(f"Data prepared successfully with shape: {df_prepared.shape}")

        model_res = ranking_model.optimize(df_prepared)
        if not model_res['success']:
            return jsonify({"error": model_res['message'], "code": 400}), 400

        heatmap_files = slotting_optimizer.generate_heatmaps_files(
            model_res['optimized_data'],
            app.config['HEATMAP_FOLDER'],
            basename=f"ranking_{uuid.uuid4()}"
        )
        
        if not heatmap_files['success']:
            app.logger.warning(f"Heatmap generation failed: {heatmap_files['message']}")
            heatmap_files = {'before_filename': None, 'after_filename': None}

        try:
            detailed_summary = slotting_optimizer.get_optimization_summary(model_res['optimized_data'])
        except AttributeError:
            detailed_summary = {
                'total_skus': len(model_res['optimized_data']),
                'zones_used': model_res['optimized_data']['zone'].nunique(),
                'total_picks': int(model_res['optimized_data']['picks'].sum())
            }

        results_payload = {
            'timestamp': datetime.now().isoformat(),
            'metrics': model_res['metrics'],
            'summary': model_res['summary'],
            'detailed_summary': detailed_summary,
            'optimized_data': model_res['optimized_data'].to_dict('records'),
            'model': 'ranking'
        }
        
        results_id = save_results_to_disk(results_payload)

        session['ranking_results'] = {
            'id': results_id, 
            'timestamp': results_payload['timestamp'],
            'model': 'ranking'
        }

        app.logger.info("=== Ranking optimization completed ===")

        response_data = {
            "message": "Ranking-based slotting optimization completed successfully",
            "optimization_metrics": model_res['metrics'],
            "optimization_summary": model_res['summary'],
            "detailed_summary": detailed_summary,
            "results_id": results_id,
            "model": "ranking",
            "data_stats": {
                "original_rows": len(df_prepared),
                "optimized_skus": len(model_res['optimized_data']),
                "total_picks": int(model_res['optimized_data']['picks'].sum()),
                "zones_used": model_res['optimized_data']['zone'].nunique()
            }
        }
        
        if heatmap_files.get('before_filename') and heatmap_files.get('after_filename'):
            response_data.update({
                "before_heatmap_url": f"/api/heatmaps/{heatmap_files['before_filename']}",
                "after_heatmap_url": f"/api/heatmaps/{heatmap_files['after_filename']}"
            })

        return jsonify(response_data)

    except Exception as e:
        app.logger.error(f"Ranking optimization error: {str(e)}\n{traceback.format_exc()}")
        return jsonify({"error": "Ranking slotting optimization failed", "code": 500}), 500

# Compare models endpoint
@app.route('/api/compare-models', methods=['POST'])
def compare_models():
    try:
        if 'user_id' not in session:
            return jsonify({"error": "Authentication required", "code": 401}), 401
        
        if 'dataset_file' not in session:
            return jsonify({"error": "No dataset found. Please upload your dataset.", "code": 400}), 400
        
        dataset_filepath = session['dataset_file']
        if not os.path.exists(dataset_filepath):
            return jsonify({"error": "Dataset file not found. Please re-upload.", "code": 400}), 400

        app.logger.info("=== Starting model comparison ===")

        has_rf_results = 'optimization_results' in session and 'id' in session['optimization_results']
        has_ranking_results = 'ranking_results' in session and 'id' in session['ranking_results']

        rf_results = None
        ranking_results = None

        if has_rf_results:
            rf_stored = load_stored_results(session['optimization_results']['id'])
            if rf_stored and rf_stored.get('model') == 'random_forest':
                rf_results = rf_stored
                app.logger.info("Using existing Random Forest results from session")

        if has_ranking_results:
            ranking_stored = load_stored_results(session['ranking_results']['id'])
            if ranking_stored and ranking_stored.get('model') == 'ranking':
                ranking_results = ranking_stored
                app.logger.info("Using existing Ranking results from session")

        df_prepared = None
        if rf_results is None or ranking_results is None:
            prep_result = data_processor.prepare_data_for_optimization(dataset_filepath)
            if not prep_result['success']:
                return jsonify({"error": prep_result['message'], "code": 400}), 400
            df_prepared = prep_result['data']

        if rf_results is None:
            app.logger.info("Running Random Forest model for comparison...")
            base_res = slotting_optimizer.optimize_slotting(df_prepared)
            if not base_res['success']:
                return jsonify({"error": f"Random Forest model failed: {base_res['message']}", "code": 400}), 400
            
            rf_results = {
                'metrics': base_res['metrics'],
                'summary': base_res['summary'],
                'model': 'random_forest'
            }

        if ranking_results is None:
            app.logger.info("Running Ranking model for comparison...")
            rank_res = ranking_model.optimize(df_prepared)
            if not rank_res['success']:
                return jsonify({"error": f"Ranking model failed: {rank_res['message']}", "code": 400}), 400
            
            ranking_results = {
                'metrics': rank_res['metrics'],
                'summary': rank_res['summary'],
                'model': 'ranking'
            }

        comparison = {
            'random_forest': {
                'metrics': rf_results['metrics'],
                'summary': rf_results['summary']
            },
            'ranking': {
                'metrics': ranking_results['metrics'],
                'summary': ranking_results['summary']
            }
        }

        def safe_float(value, default=0.0):
            try:
                return float(value) if value is not None else default
            except (ValueError, TypeError):
                return default

        rf_time_saved = safe_float(rf_results['metrics'].get('total_time_saved', 0))
        rf_time_saved_pct = safe_float(rf_results['metrics'].get('time_saved_percentage', 0))
        
        ranking_time_saved = safe_float(ranking_results['metrics'].get('total_time_saved', 0))
        ranking_time_saved_pct = safe_float(ranking_results['metrics'].get('time_saved_percentage', 0))

        comparison['performance_comparison'] = {
            'time_saved_delta_sec': round(ranking_time_saved - rf_time_saved, 1),
            'time_saved_delta_percentage': round(ranking_time_saved_pct - rf_time_saved_pct, 2),
            'winner': 'ranking' if ranking_time_saved > rf_time_saved else 'random_forest',
            'rf_used_existing': has_rf_results and load_stored_results(session['optimization_results']['id']) is not None,
            'ranking_used_existing': has_ranking_results and load_stored_results(session['ranking_results']['id']) is not None
        }

        app.logger.info("Model comparison completed successfully")

        return jsonify({
            'message': 'Model comparison completed successfully',
            'comparison': comparison
        })
        
    except Exception as e:
        app.logger.error(f"Compare models error: {str(e)}\n{traceback.format_exc()}")
        return jsonify({"error": "Failed to compare models", "code": 500}), 500

# Serve heatmap files
@app.route('/api/heatmaps/<path:filename>', methods=['GET'])
def serve_heatmap(filename):
    try:
        if not os.path.exists(os.path.join(app.config['HEATMAP_FOLDER'], filename)):
            return jsonify({"error": "Heatmap not found", "code": 404}), 404
        return send_from_directory(app.config['HEATMAP_FOLDER'], filename)
    except Exception as e:
        app.logger.error(f"Serve heatmap error: {str(e)}")
        return jsonify({"error": "Error serving heatmap", "code": 500}), 500

# FIXED DOWNLOAD ENDPOINT - Complete implementation
@app.route('/api/download-results', methods=['GET'])
def download_optimization_results():
    try:
        if 'user_id' not in session:
            return jsonify({"error": "Authentication required", "code": 401}), 401
        
        model_type = request.args.get('model', 'random_forest')
        app.logger.info(f"Download request for model: {model_type}")
        
        # Get results ID based on model type
        if model_type == 'ranking' and 'ranking_results' in session:
            results_id = session['ranking_results']['id']
            model_name = "Ranking"
        elif model_type == 'random_forest' and 'optimization_results' in session:
            results_id = session['optimization_results']['id']
            model_name = "RandomForest"
        else:
            return jsonify({"error": f"No {model_type} results found. Please run optimization first.", "code": 400}), 400

        # Load results from disk
        results_path = os.path.join(app.config['RESULTS_FOLDER'], f"{results_id}.json")
        if not os.path.exists(results_path):
            return jsonify({"error": "Results not found on server. Please rerun optimization.", "code": 400}), 400

        with open(results_path, 'r', encoding='utf-8') as f:
            stored = json.load(f)

        # Convert optimized data to DataFrame
        optimized_data_df = pd.DataFrame(stored['optimized_data'])
        
        # Create timestamp for filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        download_filename = f"slotting_results_{model_type}_{timestamp}.xlsx"
        
        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx')
        temp_filepath = temp_file.name
        temp_file.close()
        
        try:
            # Create Excel file with multiple sheets
            with pd.ExcelWriter(temp_filepath, engine='openpyxl') as writer:
                # Main results sheet - Optimized SKU assignments
                optimized_data_df.to_excel(writer, sheet_name='Optimized_Results', index=False)
                
                # Summary sheet - Key performance metrics
                summary_data = []
                for key, value in stored.get('summary', {}).items():
                    summary_data.append({'Metric': key, 'Value': value})
                if summary_data:
                    pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
                
                # Metrics sheet - Detailed optimization metrics
                metrics_data = []
                for key, value in stored.get('metrics', {}).items():
                    metrics_data.append({'Metric': key, 'Value': value})
                if metrics_data:
                    pd.DataFrame(metrics_data).to_excel(writer, sheet_name='Metrics', index=False)
                
                # Model info sheet - Metadata about the optimization run
                model_info = [
                    {'Property': 'Model Type', 'Value': stored.get('model', model_type)},
                    {'Property': 'Generated On', 'Value': stored.get('timestamp', 'Unknown')},
                    {'Property': 'Total SKUs', 'Value': len(optimized_data_df)},
                    {'Property': 'Total Records', 'Value': len(optimized_data_df)},
                    {'Property': 'Unique Zones', 'Value': optimized_data_df['zone'].nunique() if 'zone' in optimized_data_df.columns else 'N/A'},
                    {'Property': 'Data Columns', 'Value': ', '.join(optimized_data_df.columns.tolist())},
                ]
                
                # Add detailed summary if available
                if 'detailed_summary' in stored:
                    for key, value in stored['detailed_summary'].items():
                        model_info.append({'Property': f'Summary_{key}', 'Value': value})
                
                pd.DataFrame(model_info).to_excel(writer, sheet_name='Model_Info', index=False)
            
            # Clean up file after response is sent
            @after_this_request
            def cleanup_temp_file(response):
                try:
                    if os.path.exists(temp_filepath):
                        os.unlink(temp_filepath)
                        app.logger.info(f"Cleaned up temporary file: {temp_filepath}")
                except Exception as e:
                    app.logger.error(f"Error cleaning up temp file: {e}")
                return response
            
            app.logger.info(f"Sending Excel file for download: {download_filename}")
            
            return send_file(
                temp_filepath,
                as_attachment=True,
                download_name=download_filename,
                mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            )
            
        except Exception as e:
            # Clean up on error
            if os.path.exists(temp_filepath):
                os.unlink(temp_filepath)
            raise e
            
    except Exception as e:
        app.logger.error(f"Download results error: {str(e)}\n{traceback.format_exc()}")
        return jsonify({"error": "Failed to prepare download", "code": 500}), 500

# Analytics endpoints
@app.route('/api/inventory-analytics', methods=['GET'])
def get_inventory_analytics():
    try:
        if 'user_id' not in session:
            return jsonify({"error": "Authentication required", "code": 401}), 401
        
        if 'dataset_file' not in session:
            return jsonify({"error": "No dataset found. Please upload your dataset.", "code": 400}), 400
        
        dataset_filepath = session['dataset_file']
        
        if not os.path.exists(dataset_filepath):
            return jsonify({"error": "Dataset file not found. Please re-upload.", "code": 400}), 400
        
        try:
            if dataset_filepath.lower().endswith('.csv'):
                try:
                    df = pd.read_csv(dataset_filepath, encoding='utf-8')
                except UnicodeDecodeError:
                    df = pd.read_csv(dataset_filepath, encoding='latin-1')
            else:
                df = pd.read_excel(dataset_filepath)
        except Exception as e:
            return jsonify({"error": f"Failed to load dataset: {str(e)}", "code": 500}), 500
        
        try:
            analytics_result = slotting_optimizer.generate_analytics(df)
        except AttributeError:
            analytics_result = {
                'success': True,
                'analytics': {
                    'total_records': len(df),
                    'unique_skus': df['sku'].nunique() if 'sku' in df.columns else 0,
                    'unique_locations': df['location'].nunique() if 'location' in df.columns else 0,
                    'total_quantity': df['qty'].sum() if 'qty' in df.columns else 0,
                    'columns': list(df.columns)
                }
            }
        
        if not analytics_result['success']:
            return jsonify({"error": analytics_result['message'], "code": 400}), 400
        
        return jsonify({
            "message": "Inventory analytics generated successfully",
            "analytics": analytics_result['analytics']
        })
        
    except Exception as e:
        app.logger.error(f"Analytics error: {str(e)}")
        return jsonify({"error": "Analytics generation failed", "code": 500}), 500

# Data quality validation endpoint
@app.route('/api/data-quality', methods=['GET'])
def validate_data_quality():
    try:
        if 'user_id' not in session:
            return jsonify({"error": "Authentication required", "code": 401}), 401
        
        if 'dataset_file' not in session:
            return jsonify({"error": "No dataset found. Please upload your dataset.", "code": 400}), 400
        
        dataset_filepath = session['dataset_file']
        
        quality_result = data_processor.validate_data_quality(dataset_filepath)
        
        if not quality_result['success']:
            return jsonify({"error": quality_result['message'], "code": 400}), 400
        
        return jsonify({
            "message": "Data quality validation completed",
            "quality_report": quality_result['quality_report']
        })
        
    except Exception as e:
        app.logger.error(f"Data quality validation error: {str(e)}")
        return jsonify({"error": "Data quality validation failed", "code": 500}), 500

# Predict zone for new SKU (Random Forest only)
@app.route('/api/predict-sku-zone', methods=['POST'])
def predict_sku_zone():
    try:
        if 'user_id' not in session:
            return jsonify({"error": "Authentication required", "code": 401}), 401
        
        if slotting_optimizer.rf_model is None:
            return jsonify({"error": "Random Forest model not trained. Please run RF optimization first.", "code": 400}), 400
        
        data = request.get_json()
        if not data:
            return jsonify({"error": "SKU features are required", "code": 400}), 400
        
        required_features = ['picks', 'total_qty', 'total_weight', 'total_volume']
        missing_features = [f for f in required_features if f not in data]
        if missing_features:
            return jsonify({"error": f"Missing features: {missing_features}", "code": 400}), 400
        
        prediction_result = slotting_optimizer.predict_new_sku_zone(data)
        
        if not prediction_result['success']:
            return jsonify({"error": prediction_result['message'], "code": 400}), 400
        
        return jsonify({
            "message": "Zone prediction completed",
            "prediction": prediction_result
        })
        
    except Exception as e:
        app.logger.error(f"Zone prediction error: {str(e)}")
        return jsonify({"error": "Zone prediction failed", "code": 500}), 500

# Model information endpoint
@app.route('/api/models', methods=['GET'])
def get_models():
    try:
        if 'user_id' not in session:
            return jsonify({"error": "Authentication required", "code": 401}), 401
        
        models = [
            {
                "id": "warehouse_slotting_abc_fsn",
                "name": "ABC/FSN + KMeans + RandomForest Optimizer",
                "description": "Advanced warehouse slotting using ABC/FSN classification, KMeans clustering, and RandomForest prediction with enhanced location assignment",
                "version": "2.0",
                "features": [
                    "ABC/FSN classification",
                    "KMeans clustering for zone assignment", 
                    "RandomForest for future predictions",
                    "Enhanced one-SKU-per-location assignment",
                    "Interactive heatmap visualization"
                ],
                "parameters": {
                    "speed": slotting_optimizer.speed,
                    "handling_time": slotting_optimizer.handling_time,
                    "clusters": slotting_optimizer.kmeans_clusters
                },
                "endpoint": "/api/warehouse-slotting"
            },
            {
                "id": "ranking_slotting_sa",
                "name": "Ranking + KMeans + Simulated Annealing",
                "description": "Ranking-based slotting with MinMax scoring, KMeans co-occurrence clustering, and simulated annealing assignment",
                "version": "1.0",
                "features": [
                    "Composite ranking score",
                    "KMeans clustering",
                    "Simulated annealing placement",
                    "Interactive heatmaps"
                ],
                "endpoint": "/api/ranking-slotting"
            }
        ]
        return jsonify({"models": models})
        
    except Exception as e:
        app.logger.error(f"Get models error: {str(e)}")
        return jsonify({"error": "Failed to retrieve models", "code": 500}), 500

# Cache management endpoint
@app.route('/api/clear-cache', methods=['POST'])
def clear_optimization_cache():
    try:
        if 'user_id' not in session:
            return jsonify({"error": "Authentication required", "code": 401}), 401
        
        try:
            slotting_optimizer.clear_cache()
        except AttributeError:
            if hasattr(slotting_optimizer, 'cache'):
                slotting_optimizer.cache.clear()
        
        session.pop('optimization_results', None)
        session.pop('ranking_results', None)
        
        return jsonify({"message": "Optimization cache and session results cleared successfully"})
        
    except Exception as e:
        app.logger.error(f"Cache clear error: {str(e)}")
        return jsonify({"error": "Failed to clear cache", "code": 500}), 500

# Health check endpoint
@app.route('/api/health', methods=['GET'])
def health_check():
    try:
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "upload_folder_exists": os.path.exists(app.config['UPLOAD_FOLDER']),
            "heatmap_folder_exists": os.path.exists(app.config['HEATMAP_FOLDER']),
            "results_folder_exists": os.path.exists(app.config['RESULTS_FOLDER']),
            "slotting_optimizer_ready": slotting_optimizer is not None,
            "ranking_model_ready": ranking_model is not None,
            "data_processor_ready": data_processor is not None,
            "auth_manager_ready": auth_manager is not None
        }
        
        all_ready = all([
            health_status["upload_folder_exists"],
            health_status["heatmap_folder_exists"],
            health_status["results_folder_exists"],
            health_status["slotting_optimizer_ready"],
            health_status["ranking_model_ready"],
            health_status["data_processor_ready"],
            health_status["auth_manager_ready"]
        ])
        
        if not all_ready:
            health_status["status"] = "degraded"
        
        return jsonify(health_status)
        
    except Exception as e:
        app.logger.error(f"Health check error: {str(e)}")
        return jsonify({
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

# Clean up temporary files
@app.teardown_appcontext
def cleanup_temp_files(error):
    try:
        current_time = datetime.now()
        
        for folder in [app.config['UPLOAD_FOLDER'], app.config['HEATMAP_FOLDER'], app.config['RESULTS_FOLDER']]:
            if os.path.exists(folder):
                for filename in os.listdir(folder):
                    filepath = os.path.join(folder, filename)
                    if os.path.isfile(filepath):
                        try:
                            file_modified = datetime.fromtimestamp(os.path.getmtime(filepath))
                            if current_time - file_modified > timedelta(hours=24):
                                os.remove(filepath)
                                app.logger.info(f"Cleaned up old file: {filename}")
                        except Exception:
                            pass
                            
    except Exception as e:
        app.logger.error(f"Cleanup error: {str(e)}")

# Serve static files from frontend folder
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_frontend(path):
    frontend_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'frontend')
    
    if path != "" and os.path.exists(os.path.join(frontend_folder, path)):
        return send_from_directory(frontend_folder, path)
    else:
        # Changed from 'index.html' to 'home.html'
        return send_from_directory(frontend_folder, 'home.html')

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("=== Enhanced Warehouse Slotting Application Starting ===")
    print("Available endpoints:")
    print("  POST /api/signup - Create user account")
    print("  POST /api/login - User login")
    print("  POST /api/logout - User logout")
    print("  GET  /api/session - Check session")
    print("  POST /api/upload-dataset - Upload warehouse data")
    print("  GET  /api/dataset-preview - Preview uploaded data")
    print("  POST /api/warehouse-slotting - Run Random Forest optimization")
    print("  POST /api/ranking-slotting - Run Ranking model optimization")
    print("  POST /api/compare-models - Compare both models")
    print("  GET  /api/heatmaps/<filename> - Serve heatmap files")
    print("  GET  /api/inventory-analytics - Get inventory analytics")
    print("  GET  /api/data-quality - Validate data quality")
    print("  GET  /api/download-results - Download Excel results (FIXED)")
    print("  POST /api/predict-sku-zone - Predict zone for new SKU (RF only)")
    print("  GET  /api/models - Get available models")
    print("  POST /api/clear-cache - Clear optimization cache")
    print("  GET  /api/health - Health check")
    print("=== Application Ready with Fixed Download Functionality ===")
    
    app.run(debug=True, host='0.0.0.0', port=5000)