import hashlib
import uuid
from datetime import datetime
import os
import json

class AuthManager:
    """Simple session-based authentication manager."""
    
    def __init__(self):
        self.users_file = 'users.json'
        self.users = self._load_users()
    
    def _load_users(self) -> dict:
        """Load users from JSON file."""
        try:
            if os.path.exists(self.users_file):
                with open(self.users_file, 'r') as f:
                    return json.load(f)
            else:
                return {}
        except Exception as e:
            print(f"Error loading users: {str(e)}")
            return {}
    
    def _save_users(self) -> bool:
        """Save users to JSON file."""
        try:
            with open(self.users_file, 'w') as f:
                json.dump(self.users, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving users: {str(e)}")
            return False
    
    def _hash_password(self, password: str, salt: str = None) -> tuple:
        """Hash password with salt."""
        try:
            if salt is None:
                salt = uuid.uuid4().hex
            
            # Combine password and salt
            password_hash = hashlib.sha256((password + salt).encode()).hexdigest()
            return password_hash, salt
        except Exception as e:
            raise Exception(f"Error hashing password: {str(e)}")
    
    def _verify_password(self, password: str, stored_hash: str, salt: str) -> bool:
        """Verify password against stored hash."""
        try:
            password_hash, _ = self._hash_password(password, salt)
            return password_hash == stored_hash
        except Exception as e:
            return False
    
    def create_user(self, username: str, password: str) -> dict:
        """Create a new user account."""
        try:
            # Validate input
            if not username or not password:
                return {"success": False, "message": "Username and password are required"}
            
            if len(username) < 3:
                return {"success": False, "message": "Username must be at least 3 characters long"}
            
            if len(password) < 6:
                return {"success": False, "message": "Password must be at least 6 characters long"}
            
            # Check if username already exists
            if username.lower() in self.users:
                return {"success": False, "message": "Username already exists"}
            
            # Hash password
            password_hash, salt = self._hash_password(password)
            
            # Create user record
            user_id = str(uuid.uuid4())
            user_data = {
                "user_id": user_id,
                "username": username,
                "password_hash": password_hash,
                "salt": salt,
                "created_at": datetime.now().isoformat(),
                "last_login": None,
                "login_count": 0
            }
            
            # Store user
            self.users[username.lower()] = user_data
            
            # Save to file
            if not self._save_users():
                return {"success": False, "message": "Failed to save user data"}
            
            return {
                "success": True, 
                "message": "User created successfully",
                "user_id": user_id
            }
            
        except Exception as e:
            return {"success": False, "message": f"Error creating user: {str(e)}"}
    
    def authenticate_user(self, username: str, password: str) -> dict:
        """Authenticate user credentials."""
        try:
            # Validate input
            if not username or not password:
                return {"success": False, "message": "Username and password are required"}
            
            # Check if user exists
            username_lower = username.lower()
            if username_lower not in self.users:
                return {"success": False, "message": "Invalid credentials"}
            
            user_data = self.users[username_lower]
            
            # Verify password
            if not self._verify_password(password, user_data["password_hash"], user_data["salt"]):
                return {"success": False, "message": "Invalid credentials"}
            
            # Update login information
            user_data["last_login"] = datetime.now().isoformat()
            user_data["login_count"] = user_data.get("login_count", 0) + 1
            
            # Save updated user data
            self._save_users()
            
            return {
                "success": True,
                "message": "Authentication successful",
                "user_id": user_data["user_id"],
                "username": user_data["username"]
            }
            
        except Exception as e:
            return {"success": False, "message": f"Authentication error: {str(e)}"}
    
    def get_user_info(self, user_id: str) -> dict:
        """Get user information by user ID."""
        try:
            for username, user_data in self.users.items():
                if user_data["user_id"] == user_id:
                    return {
                        "success": True,
                        "user_info": {
                            "user_id": user_data["user_id"],
                            "username": user_data["username"],
                            "created_at": user_data["created_at"],
                            "last_login": user_data["last_login"],
                            "login_count": user_data.get("login_count", 0)
                        }
                    }
            
            return {"success": False, "message": "User not found"}
            
        except Exception as e:
            return {"success": False, "message": f"Error retrieving user info: {str(e)}"}
    
    def change_password(self, user_id: str, old_password: str, new_password: str) -> dict:
        """Change user password."""
        try:
            # Validate input
            if not old_password or not new_password:
                return {"success": False, "message": "Old and new passwords are required"}
            
            if len(new_password) < 6:
                return {"success": False, "message": "New password must be at least 6 characters long"}
            
            # Find user
            user_data = None
            username_key = None
            for username, data in self.users.items():
                if data["user_id"] == user_id:
                    user_data = data
                    username_key = username
                    break
            
            if not user_data:
                return {"success": False, "message": "User not found"}
            
            # Verify old password
            if not self._verify_password(old_password, user_data["password_hash"], user_data["salt"]):
                return {"success": False, "message": "Invalid current password"}
            
            # Hash new password
            new_password_hash, new_salt = self._hash_password(new_password)
            
            # Update user data
            user_data["password_hash"] = new_password_hash
            user_data["salt"] = new_salt
            user_data["password_changed_at"] = datetime.now().isoformat()
            
            # Save to file
            if not self._save_users():
                return {"success": False, "message": "Failed to save password change"}
            
            return {"success": True, "message": "Password changed successfully"}
            
        except Exception as e:
            return {"success": False, "message": f"Error changing password: {str(e)}"}
    
    def delete_user(self, user_id: str) -> dict:
        """Delete user account."""
        try:
            # Find and remove user
            username_to_delete = None
            for username, user_data in self.users.items():
                if user_data["user_id"] == user_id:
                    username_to_delete = username
                    break
            
            if not username_to_delete:
                return {"success": False, "message": "User not found"}
            
            # Delete user
            del self.users[username_to_delete]
            
            # Save changes
            if not self._save_users():
                return {"success": False, "message": "Failed to delete user"}
            
            return {"success": True, "message": "User deleted successfully"}
            
        except Exception as e:
            return {"success": False, "message": f"Error deleting user: {str(e)}"}
    
    def list_users(self) -> dict:
        """List all users (admin function)."""
        try:
            user_list = []
            for username, user_data in self.users.items():
                user_list.append({
                    "username": user_data["username"],
                    "created_at": user_data["created_at"],
                    "last_login": user_data["last_login"],
                    "login_count": user_data.get("login_count", 0)
                })
            
            return {"success": True, "users": user_list, "total_users": len(user_list)}
            
        except Exception as e:
            return {"success": False, "message": f"Error listing users: {str(e)}"}
    
    def cleanup_old_sessions(self, days: int = 30) -> dict:
        """Clean up old user sessions (placeholder for session management)."""
        try:
            # This is a placeholder for session cleanup
            # In a production environment, you would implement proper session management
            return {"success": True, "message": f"Session cleanup completed for sessions older than {days} days"}
            
        except Exception as e:
            return {"success": False, "message": f"Error during cleanup: {str(e)}"}
    
    def validate_session(self, user_id: str) -> dict:
        """Validate if user session is still valid."""
        try:
            # Check if user exists
            for username, user_data in self.users.items():
                if user_data["user_id"] == user_id:
                    return {
                        "success": True,
                        "valid": True,
                        "username": user_data["username"]
                    }
            
            return {"success": True, "valid": False, "message": "Invalid session"}
            
        except Exception as e:
            return {"success": False, "message": f"Session validation error: {str(e)}"}