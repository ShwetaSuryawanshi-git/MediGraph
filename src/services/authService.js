// File: MediGraph/src/services/authService.js
// Path: /Users/shweta/Desktop/major project fy/MediGraph/src/services/authService.js

/**
 * Authentication Service for MediGraph
 * Handles user signup, login, logout, and auth state management
 */

import {
    createUserWithEmailAndPassword,
    signInWithEmailAndPassword,
    signOut,
    onAuthStateChanged
} from 'firebase/auth';
import { doc, setDoc, updateDoc, serverTimestamp } from 'firebase/firestore';
import { auth, db } from '../../firebase/firebase';

/**
 * Sign up a new user with email and password
 * Creates a corresponding user document in Firestore
 * 
 * @param {string} email - User's email address
 * @param {string} password - User's password (min 6 characters)
 * @param {string} displayName - Optional display name
 * @returns {Promise<Object>} - User credential object
 */
export const signUp = async (email, password, displayName = null) => {
    try {
        // Create Firebase Auth user
        const userCredential = await createUserWithEmailAndPassword(auth, email, password);
        const user = userCredential.user;

        // Create Firestore user document
        // Document ID = uid (as per schema contract)
        await setDoc(doc(db, 'users', user.uid), {
            email: user.email,
            display_name: displayName,
            role: 'researcher', // Default role as per security rules
            created_at: serverTimestamp(),
            last_login_at: serverTimestamp()
        });

        return userCredential;
    } catch (error) {
        console.error('Signup error:', error);
        throw error;
    }
};

/**
 * Sign in an existing user with email and password
 * Updates last_login_at timestamp in Firestore
 * 
 * @param {string} email - User's email address
 * @param {string} password - User's password
 * @returns {Promise<Object>} - User credential object
 */
export const signIn = async (email, password) => {
    try {
        const userCredential = await signInWithEmailAndPassword(auth, email, password);
        const user = userCredential.user;

        // Update last login timestamp
        await updateDoc(doc(db, 'users', user.uid), {
            last_login_at: serverTimestamp()
        });

        return userCredential;
    } catch (error) {
        console.error('Sign in error:', error);
        throw error;
    }
};

/**
 * Sign out the current user
 * 
 * @returns {Promise<void>}
 */
export const logOut = async () => {
    try {
        await signOut(auth);
    } catch (error) {
        console.error('Logout error:', error);
        throw error;
    }
};

/**
 * Subscribe to authentication state changes
 * 
 * @param {Function} callback - Callback function receiving user object (or null)
 * @returns {Function} - Unsubscribe function
 */
export const subscribeToAuthState = (callback) => {
    return onAuthStateChanged(auth, (user) => {
        callback(user);
    });
};

/**
 * Get the current authenticated user
 * 
 * @returns {Object|null} - Current user or null if not authenticated
 */
export const getCurrentUser = () => {
    return auth.currentUser;
};
