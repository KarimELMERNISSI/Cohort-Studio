"""
Users Management Page - Admin Only

This page allows administrators to:
- View all registered users with their status
- Activate/Deactivate user accounts
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
        st.error("Access Denied. Admin privileges required.")
        st.stop()
    
    st.markdown("Manage user accounts - activate, reset passwords, or delete users.")
    
    # Get DB Manager
    if 'db_manager' not in st.session_state:
        st.session_state.db_manager = DBManager()
    db = st.session_state.db_manager
    
    # Force re-initialization to ensure migration runs
    db.init_user_db()
    
    # Fetch all users with status
    users = db.get_all_users_with_status()
    
    # Fallback to simple user list if status query fails
    if not users:
        simple_users = db.get_all_users()
        if simple_users:
            # Convert to status format with defaults
            users = [{'username': u, 'is_active': True, 'status': 'Active', 'created_at': None} for u in simple_users]
    
    if not users:
        st.info("No users registered yet.")
        return
    
    # Count pending users
    pending_count = sum(1 for u in users if u['status'] == 'Pending Approval')
    
    if pending_count > 0:
        st.warning(f"**{pending_count} user(s) awaiting approval**")
    
    # --- Users Table ---
    st.subheader("Registered Users")
    
    # Header
    cols = st.columns([2, 2, 2, 2])
    with cols[0]:
        st.markdown("**Username**")
    with cols[1]:
        st.markdown("**Status**")
    with cols[2]:
        st.markdown("**Activation**")
    with cols[3]:
        st.markdown("**Actions**")
    
    st.divider()
    
    for user in users:
        uname = user['username']
        status = user['status']
        is_active = user['is_active']
        
        cols = st.columns([2, 2, 2, 2])
        
        with cols[0]:
            if uname == 'admin':
                st.markdown(f"👑 **{uname}**")
            else:
                st.markdown(f"👤 {uname}")
        
        with cols[1]:
            if status == 'Active':
                st.markdown("🟢 **Active**")
            else:
                st.markdown("🟡 **Pending**")
        
        with cols[2]:
            if uname == 'admin':
                st.markdown("—")  # Admin is always active
            elif is_active:
                if st.button("Deactivate", key=f"deact_{uname}", help=f"Deactivate {uname}"):
                    success, msg = db.deactivate_user(uname)
                    if success:
                        st.toast(msg, icon="")
                        st.rerun()
                    else:
                        st.error(msg)
            else:
                if st.button("Activate", key=f"act_{uname}", help=f"Approve {uname}", type="primary"):
                    success, msg = db.activate_user(uname)
                    if success:
                        st.toast(msg, icon="")
                        st.rerun()
                    else:
                        st.error(msg)
        
        with cols[3]:
            if uname != 'admin':
                if st.button("", key=f"delete_{uname}", help=f"Delete user {uname}"):
                    st.session_state[f"confirm_delete_{uname}"] = True
    
    # --- Delete Confirmation Dialogs ---
    for user in users:
        uname = user['username']
        if st.session_state.get(f"confirm_delete_{uname}", False):
            st.warning(f"Are you sure you want to delete user **{uname}**?")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Yes, Delete", key=f"confirm_yes_{uname}"):
                    success, msg = db.delete_user(uname)
                    if success:
                        st.success(msg)
                        st.session_state[f"confirm_delete_{uname}"] = False
                        st.rerun()
                    else:
                        st.error(msg)
            with col2:
                if st.button("Cancel", key=f"confirm_no_{uname}"):
                    st.session_state[f"confirm_delete_{uname}"] = False
                    st.rerun()
    
    st.divider()
    
    # --- Password Reset Section ---
    st.subheader("Reset User Password")
    
    # Filter out admin from password reset
    non_admin_users = [u['username'] for u in users if u['username'] != 'admin']
    
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
            
            submit = st.form_submit_button("Reset Password", width="stretch")
            
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
        **User Activation:**
        - New users start with **Pending Approval** status
        - Users cannot log in until admin activates their account
        - Admin account is always active
        
        **Security Notes:**
        - Passwords are stored using **bcrypt** hashing (one-way encryption)
        - Original passwords **cannot be retrieved** - only reset
        - The **admin** account cannot be deleted or deactivated
        """)
