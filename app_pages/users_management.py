"""
Users Management Page - Admin Only

This page allows administrators to:
- View all registered users
- Reset user passwords
- Delete user accounts
"""

import streamlit as st
from manage.db_manager import DBManager


def app():
    """Render the Users Management page."""
    
    # Double-check admin access (should already be filtered in main.py)
    username = st.session_state.get('username')
    if username != 'admin':
        st.error("⛔ Access Denied. Admin privileges required.")
        st.stop()
    
    st.title("👥 Users Management")
    st.markdown("Manage user accounts - view, reset passwords, or delete users.")
    
    # Get DB Manager
    if 'db_manager' not in st.session_state:
        st.session_state.db_manager = DBManager()
    db = st.session_state.db_manager
    
    # Fetch all users
    users = db.get_all_users()
    
    if not users:
        st.info("No users registered yet.")
        return
    
    # --- Users Table ---
    st.subheader("📋 Registered Users")
    
    # Create a simple table display
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("**Username**")
    with col2:
        st.markdown("**Actions**")
    
    st.divider()
    
    for user in users:
        col1, col2 = st.columns([3, 1])
        with col1:
            if user == 'admin':
                st.markdown(f"👑 **{user}** (Admin)")
            else:
                st.markdown(f"👤 {user}")
        with col2:
            if user != 'admin':
                if st.button("🗑️", key=f"delete_{user}", help=f"Delete user {user}"):
                    st.session_state[f"confirm_delete_{user}"] = True
    
    # --- Delete Confirmation Dialogs ---
    for user in users:
        if st.session_state.get(f"confirm_delete_{user}", False):
            st.warning(f"⚠️ Are you sure you want to delete user **{user}**?")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("✅ Yes, Delete", key=f"confirm_yes_{user}"):
                    success, msg = db.delete_user(user)
                    if success:
                        st.success(msg)
                        st.session_state[f"confirm_delete_{user}"] = False
                        st.rerun()
                    else:
                        st.error(msg)
            with col2:
                if st.button("❌ Cancel", key=f"confirm_no_{user}"):
                    st.session_state[f"confirm_delete_{user}"] = False
                    st.rerun()
    
    st.divider()
    
    # --- Password Reset Section ---
    st.subheader("🔐 Reset User Password")
    
    # Filter out admin from password reset (admin can reset their own from login)
    non_admin_users = [u for u in users if u != 'admin']
    
    if not non_admin_users:
        st.info("No non-admin users to manage.")
    else:
        with st.form("reset_password_form"):
            selected_user = st.selectbox(
                "Select User",
                options=non_admin_users,
                help="Select the user whose password you want to reset"
            )
            
            new_password = st.text_input(
                "New Password",
                type="password",
                help="Enter the new password (minimum 4 characters)"
            )
            
            confirm_password = st.text_input(
                "Confirm Password",
                type="password",
                help="Re-enter the new password to confirm"
            )
            
            submit = st.form_submit_button("🔄 Reset Password", use_container_width=True)
            
            if submit:
                if not new_password:
                    st.error("Password cannot be empty.")
                elif len(new_password) < 4:
                    st.error("Password must be at least 4 characters.")
                elif new_password != confirm_password:
                    st.error("Passwords do not match.")
                else:
                    success, msg = db.update_user_password(selected_user, new_password)
                    if success:
                        st.success(msg)
                    else:
                        st.error(msg)
    
    # --- Admin Info ---
    st.divider()
    with st.expander("ℹ️ About User Management"):
        st.markdown("""
        **Security Notes:**
        - Passwords are stored using **bcrypt** hashing (one-way encryption)
        - Original passwords **cannot be retrieved** - only reset
        - The **admin** account cannot be deleted
        - Users can also change their own credentials via the login page
        """)
