// File: MediGraph/src/components/Auth/SignupForm.jsx
// Path: /Users/shweta/Desktop/major project fy/MediGraph/src/components/Auth/SignupForm.jsx

/**
 * Signup Form Component
 * Handles new user registration
 */

import React, { useState } from 'react';
import { useAuth } from '../../hooks/useAuth';

const SignupForm = ({ onSwitchToLogin }) => {
    const { signUp, loading, error, clearError } = useAuth();

    const [displayName, setDisplayName] = useState('');
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [confirmPassword, setConfirmPassword] = useState('');
    const [localError, setLocalError] = useState('');

    const handleSubmit = async (e) => {
        e.preventDefault();
        setLocalError('');
        clearError();

        // Validation
        if (!email || !password) {
            setLocalError('Email and password are required');
            return;
        }

        if (password.length < 6) {
            setLocalError('Password must be at least 6 characters');
            return;
        }

        if (password !== confirmPassword) {
            setLocalError('Passwords do not match');
            return;
        }

        try {
            await signUp(email, password, displayName || null);
            // Navigation handled by auth state change
        } catch (err) {
            // Error already set in auth context
        }
    };

    const displayError = localError || error;

    return (
        <div className="glass-card animate-slide-up" style={{ maxWidth: '400px', width: '100%' }}>
            <div style={{ textAlign: 'center', marginBottom: 'var(--spacing-xl)' }}>
                <h2 style={{ marginBottom: 'var(--spacing-sm)' }}>Create Account</h2>
                <p className="text-secondary">Join MediGraph as a researcher</p>
            </div>

            <form onSubmit={handleSubmit}>
                <div className="form-group">
                    <label className="form-label" htmlFor="displayName">
                        Display Name <span className="text-muted">(optional)</span>
                    </label>
                    <input
                        id="displayName"
                        type="text"
                        className="form-input"
                        placeholder="Enter your name"
                        value={displayName}
                        onChange={(e) => setDisplayName(e.target.value)}
                        disabled={loading}
                        autoComplete="name"
                    />
                </div>

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
                        placeholder="Min 6 characters"
                        value={password}
                        onChange={(e) => setPassword(e.target.value)}
                        disabled={loading}
                        autoComplete="new-password"
                    />
                </div>

                <div className="form-group">
                    <label className="form-label" htmlFor="confirmPassword">Confirm Password</label>
                    <input
                        id="confirmPassword"
                        type="password"
                        className="form-input"
                        placeholder="Confirm your password"
                        value={confirmPassword}
                        onChange={(e) => setConfirmPassword(e.target.value)}
                        disabled={loading}
                        autoComplete="new-password"
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
                    {loading ? 'Creating account...' : 'Create Account'}
                </button>
            </form>

            <div style={{ textAlign: 'center' }}>
                <span className="text-secondary">Already have an account? </span>
                <button
                    type="button"
                    className="btn btn-ghost"
                    onClick={onSwitchToLogin}
                    style={{ padding: 0 }}
                >
                    Sign in
                </button>
            </div>
        </div>
    );
};

export default SignupForm;
