// File: MediGraph/src/components/Auth/AuthPage.jsx
// Path: /Users/shweta/Desktop/major project fy/MediGraph/src/components/Auth/AuthPage.jsx

/**
 * Authentication Page Component
 * Switches between Login and Signup forms
 */

import React, { useState } from 'react';
import LoginForm from './LoginForm';
import SignupForm from './SignupForm';

const AuthPage = () => {
    const [isLogin, setIsLogin] = useState(true);

    return (
        <div
            style={{
                minHeight: '100vh',
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                padding: 'var(--spacing-lg)'
            }}
        >
            {/* Logo / Brand */}
            <div style={{ marginBottom: 'var(--spacing-xl)', textAlign: 'center' }}>
                <h1 style={{
                    fontSize: '2.5rem',
                    background: 'linear-gradient(135deg, var(--color-primary), var(--color-disease))',
                    WebkitBackgroundClip: 'text',
                    WebkitTextFillColor: 'transparent',
                    marginBottom: 'var(--spacing-sm)'
                }}>
                    🧬 MediGraph
                </h1>
                <p className="text-secondary">
                    Drug-Target Interaction & Repurposing Platform
                </p>
            </div>

            {/* Auth Forms */}
            {isLogin ? (
                <LoginForm onSwitchToSignup={() => setIsLogin(false)} />
            ) : (
                <SignupForm onSwitchToLogin={() => setIsLogin(true)} />
            )}
        </div>
    );
};

export default AuthPage;
