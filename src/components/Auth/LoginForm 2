// File: MediGraph/src/components/Auth/LoginForm.jsx
// Path: /Users/shweta/Desktop/major project fy/MediGraph/src/components/Auth/LoginForm.jsx

/**
 * Login Form Component
 * Handles user authentication via email/password
 */

import React, { useState } from 'react';
import { useAuth } from '../../hooks/useAuth';

const LoginForm = ({ onSwitchToSignup }) => {
    const { signIn, loading, error, clearError } = useAuth();

    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [localError, setLocalError] = useState('');

    const handleSubmit = async (e) => {
        e.preventDefault();
        setLocalError('');
        clearError();

        // Basic validation
        if (!email || !password) {
            setLocalError('Please fill in all fields');
            return;
        }

        try {
            await signIn(email, password);
            // Navigation handled by auth state change
        } catch (err) {
            // Error already set in auth context
        }
    };

    const displayError = localError || error;

    return (
        <div className="glass-card animate-slide-up" style={{ maxWidth: '400px', width: '100%' }}>
            <div style={{ textAlign: 'center', marginBottom: 'var(--spacing-xl)' }}>
                <h2 style={{ marginBottom: 'var(--spacing-sm)' }}>Welcome Back</h2>
                <p className="text-secondary">Sign in to MediGraph</p>
            </div>

            <form onSubmit={handleSubmit}>
                <div className="form-group">
                    <label className="form-label" htmlFor="email">Email</label>
                    <input
                        id="email"
                        type="email"
                        className="form-input"
                        placeholder="Enter your email"
                        value={email}
                        onChange={(e) => setEmail(e.target.value)}
                        disabled={loading}
                        autoComplete="email"
                    />
                </div>

                <div className="form-group">
                    <label className="form-label" htmlFor="password">Password</label>
                    <input
                        id="password"
                        type="password"
                        className="form-input"
                        placeholder="Enter your password"
                        value={password}
                        onChange={(e) => setPassword(e.target.value)}
                        disabled={loading}
                        autoComplete="current-password"
                    />
                </div>

                {displayError && (
                    <div className="form-error" style={{ marginBottom: 'var(--spacing-md)' }}>
                        {displayError}
                    </div>
                )}

                <button
                    type="submit"
                    className="btn btn-primary"
                    disabled={loading}
                    style={{ width: '100%', marginBottom: 'var(--spacing-md)' }}
                >
                    {loading ? 'Signing in...' : 'Sign In'}
                </button>
            </form>

            <div style={{ textAlign: 'center' }}>
                <span className="text-secondary">Don't have an account? </span>
                <button
                    type="button"
                    className="btn btn-ghost"
                    onClick={onSwitchToSignup}
                    style={{ padding: 0 }}
                >
                    Sign up
                </button>
            </div>
        </div>
    );
};

export default LoginForm;
