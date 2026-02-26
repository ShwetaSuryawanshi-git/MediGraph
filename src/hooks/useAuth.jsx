// File: MediGraph/src/hooks/useAuth.js
// Path: /Users/shweta/Desktop/major project fy/MediGraph/src/hooks/useAuth.js

/**
 * Custom React Hook for Authentication State Management
 * Provides auth state, loading status, and auth methods to components
 */

import { useState, useEffect, useCallback, createContext, useContext } from 'react';
import {
    signUp,
    signIn,
    logOut,
    subscribeToAuthState,
    getCurrentUser
} from '../services/authService';

// Create Auth Context
const AuthContext = createContext(null);

/**
 * Auth Provider Component
 * Wraps the application and provides auth state to all child components
 */
export const AuthProvider = ({ children }) => {
    const [user, setUser] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    // Subscribe to auth state changes on mount
    useEffect(() => {
        const unsubscribe = subscribeToAuthState((authUser) => {
            setUser(authUser);
            setLoading(false);
        });

        // Cleanup subscription on unmount
        return () => unsubscribe();
    }, []);

    // Sign up handler
    const handleSignUp = useCallback(async (email, password, displayName) => {
        setError(null);
        setLoading(true);
        try {
            const userCredential = await signUp(email, password, displayName);
            return userCredential;
        } catch (err) {
            setError(err.message);
            throw err;
        } finally {
            setLoading(false);
        }
    }, []);

    // Sign in handler
    const handleSignIn = useCallback(async (email, password) => {
        setError(null);
        setLoading(true);
        try {
            const userCredential = await signIn(email, password);
            return userCredential;
        } catch (err) {
            setError(err.message);
            throw err;
        } finally {
            setLoading(false);
        }
    }, []);

    // Logout handler
    const handleLogout = useCallback(async () => {
        setError(null);
        try {
            await logOut();
        } catch (err) {
            setError(err.message);
            throw err;
        }
    }, []);

    // Clear error
    const clearError = useCallback(() => {
        setError(null);
    }, []);

    const value = {
        user,
        loading,
        error,
        isAuthenticated: !!user,
        signUp: handleSignUp,
        signIn: handleSignIn,
        logout: handleLogout,
        clearError
    };

    return (
        <AuthContext.Provider value={value}>
            {children}
        </AuthContext.Provider>
    );
};

/**
 * Custom hook to access auth context
 * Must be used within AuthProvider
 */
export const useAuth = () => {
    const context = useContext(AuthContext);
    if (!context) {
        throw new Error('useAuth must be used within an AuthProvider');
    }
    return context;
};

export default useAuth;
