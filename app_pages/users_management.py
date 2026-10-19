"""
Users page (Admin > Users), for administrators only

This page allows administrators to:
- View all registered users with their status
- Activate/Deactivate user accounts
- Reset user passwords
- Delete user accounts
"""

import streamlit as st

from manage.iam import IAMProviderType, get_iam_manager
from manage.iam.authz import MIN_PASSWORD_LENGTH, is_admin_role

PROVIDER_LABELS = {
    IAMProviderType.LOCAL: "local accounts (this server)",
    IAMProviderType.BETTER_AUTH: "Better Auth (managed through its admin API)",
}


def app():
    """Render the Users page."""

    # Double-check admin access (should already be filtered in main.py)
    if not is_admin_role(st.session_state.get('user_role')):
        st.error("Access Denied. Admin privileges required.")
        st.stop()

    st.markdown("Manage user accounts - activate, reset passwords, or delete users.")

    # The accounts of the identity provider that signs users in (CMP_IAM_PROVIDER): this page managed the local
    # accounts whatever the provider, so under Better Auth its changes reached nobody who signs in
    db = get_iam_manager()
    provider = db.provider_type
    if provider == IAMProviderType.DISABLED:
        st.info("Sign-in is disabled on this server (CMP_IAM_PROVIDER=disabled): there are no accounts to manage.")
        return
    st.caption(f"Accounts of the identity provider: {PROVIDER_LABELS.get(provider, provider.value)}.")

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
            if is_admin_role(user.get('role')):
                st.markdown(f"**{uname}** `(Admin)`")
            else:
                st.markdown(f"**{uname}**")
        
        with cols[1]:
            if status == 'Active':
                st.markdown(":green[**Active**]")
            else:
                st.markdown(":orange[**Pending**]")
        
        with cols[2]:
            if is_admin_role(user.get('role')):
                st.markdown("—")  # administrator accounts cannot be deactivated
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
            if not is_admin_role(user.get('role')):
                if st.button("Delete", key=f"delete_{uname}", help=f"Delete user {uname}"):
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
    non_admin_users = [u['username'] for u in users if not is_admin_role(u.get('role'))]
    
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
                help=f"Enter the new password (minimum {MIN_PASSWORD_LENGTH} characters)"
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
                elif len(new_password) < MIN_PASSWORD_LENGTH:
                    st.error(f"Password must be at least {MIN_PASSWORD_LENGTH} characters.")
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
    with st.expander("About User Management"):
        st.markdown("""
        **User Activation:**
        - New users start with **Pending Approval** status
        - Users cannot log in until admin activates their account
        - Administrator accounts are created with `python manage_users.py create-admin <name>` or CMP_ADMIN_* settings
        
        **Security Notes:**
        - Passwords are stored using **bcrypt** hashing (one-way encryption)
        - Original passwords **cannot be retrieved** - only reset
        - Administrator accounts cannot be deleted or deactivated
        """)
